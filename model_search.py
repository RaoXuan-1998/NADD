import torch
import torch.nn as nn
from operations import *
from genotypes import PRIMITIVES
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class Normal_Computing_Node(nn.Module):
    def __init__(self, c_now, node_id, primitives = PRIMITIVES,
                 affine = False, baseline = 0.01, channel_shuffle = False, k = 1):
        
        super(Normal_Computing_Node, self).__init__()
        self.baseline = baseline
        self.c_now = c_now
        self.node_id = node_id
        self.primitives = primitives
        self.ops = nn.ModuleList()
        self.indicator = []
        self.channel_shuffle = channel_shuffle
        
        if channel_shuffle == True:
            self.k = k
            for input_order in range(node_id):
                for op_id, op_name in enumerate(primitives):
                    self.indicator.append(input_order)
                    op = OPS[op_name](int(c_now // self.k), 1, affine)
                    self.ops.append(op)
        
        else:    
            for input_order in range(node_id):
                for op_id, op_name in enumerate(primitives):
                    self.indicator.append(input_order)
                    op = OPS[op_name](c_now, 1, affine)
                    self.ops.append(op)
    
    def forward(self, states, weights):
        if not self.channel_shuffle:
            state = 0.0
            for order, op in enumerate(self.ops):
                input_order = self.indicator[order]
                input_state = states[input_order]
                w = weights[order]
                if not w < self.baseline:
                    state = state + weights[order]*op(input_state)                
            return state
        else:
            state = 0.0
            for order, op in enumerate(self.ops):
                input_order = self.indicator[order]
                input_state = states[input_order]
                dim_2 = input_state.shape[1]
                forward_part = input_state[:, 0:int(dim_2//self.k), :, :]
                residual_part = input_state[:, int(dim_2//self.k):, :, :]
                w = weights[order]
                if not w < self.baseline:
                    output_part = weights[order]*op(forward_part)
                    state_ = torch.cat([residual_part, output_part], dim = 1)
                    state = state + state_
            
            return channel_shuffle(state, self.k)

    
class Reduction_Computing_Node(nn.Module):
    def __init__(self, c_now, node_id, primitives=PRIMITIVES,
                 affine = False, baseline = 0.01, channel_shuffle = False, k = 1):
        super(Reduction_Computing_Node, self).__init__()
        self.baseline = baseline
        self.c_now = c_now
        self.node_id = node_id
        self.primitives = []
        self.ops = nn.ModuleList()
        self.indicator = []
        
        self.channel_shuffle = channel_shuffle
        
        if channel_shuffle == True:
            self.k = k
            self.mp = nn.MaxPool2d(2,2)
            for input_order in range(node_id):
                if input_order < 2:
                    stride = 2
                else:
                    stride = 1                                
                for op_id, op_name in enumerate(primitives):
                    self.indicator.append(input_order)
                    op = OPS[op_name](int(c_now // self.k), stride, affine)
                    self.ops.append(op)
        else:
            for input_order in range(node_id):
                if input_order < 2:
                    stride = 2
                else:
                    stride = 1
                for op_id, op_name in enumerate(primitives):
                    self.indicator.append(input_order)
                    self.primitives.append(op_name)
                    op = OPS[op_name](c_now, stride, affine)
                    self.ops.append(op)
    
    def forward(self, states, weights):
        if not self.channel_shuffle:
            state = 0.0
            for order, op in enumerate(self.ops):
                input_order = self.indicator[order]
                input_state = states[input_order]
                w = weights[order]
                if not w < self.baseline:
                    state = state + weights[order]*op(input_state)                
            return state
        else:
            state = 0.0
            for order, op in enumerate(self.ops):
                input_order = self.indicator[order]
                input_state = states[input_order]
                dim_2 = input_state.shape[1]
                forward_part = input_state[:, 0:int(dim_2//self.k), :, :]
                residual_part = input_state[:, int(dim_2//self.k): , :, :]
                w = weights[order]
                if not w < self.baseline:
                    output_part = weights[order]*op(forward_part)
                    if input_order < 2:
                        state_ = torch.cat([self.mp(residual_part), output_part], dim = 1)
                    else:
                        state_ = torch.cat([residual_part, output_part], dim = 1)
                    state = state + state_
            
            return channel_shuffle(state, self.k)

    
class Cell(nn.Module):
    def __init__(self, c_now,reduction_now, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,
                 cell_id, primitives, computing_node_num = 4, concat_num = 4, affine = False, baseline = 0.01,
                 channel_shuffle = False, k = 1):
        super(Cell,self).__init__()
        self.baseline = baseline
        self.c_now = c_now
        self.c_prev_0 = c_prev_0
        self.c_prev_1 = c_prev_1
        self.cell_id = cell_id
        self.reduction = reduction_now
        self.reduction_prev_0 = reduction_prev_0
        self.reduction_prev_1 = reduction_prev_1  
        self.concat_num = concat_num
        self.computing_node_num = computing_node_num
        self.preprocess_0 = ReLUConvBN(c_prev_0, c_now, kernel_size=1, stride=1, padding=0, affine=False)
        if reduction_prev_0:            
            self.preprocess_1 = ReduceOperator(c_prev_1, c_now, affine=False)
        else:
            self.preprocess_1 = ReLUConvBN(c_prev_1, c_now, kernel_size=1, stride=1,padding=0, affine=False)
        self.computing_nodes = nn.ModuleList()
        self.channel_shuffle = channel_shuffle
        self.k = k
        
        for computing_node_order in range(computing_node_num):
            if reduction_now:
                computing_node = Reduction_Computing_Node(c_now, computing_node_order + 2, primitives, affine, baseline, channel_shuffle, k)
            else:
                computing_node = Normal_Computing_Node(c_now, computing_node_order + 2, primitives, affine, baseline, channel_shuffle, k)
            self.computing_nodes.append(computing_node)
    
    def forward(self, input_prev_0, input_prev_1, architecture):
        state_prev_0 = self.preprocess_0(input_prev_0)
        state_prev_1 = self.preprocess_1(input_prev_1)
        states = [state_prev_0, state_prev_1]
        for computing_node_order in range(self.computing_node_num):
            node_name = 'cell_{},node_{}'.format(self.cell_id, computing_node_order + 2)
            node_weights = architecture.get_weights(node_name, forward = True)
            state = self.computing_nodes[computing_node_order](states, node_weights)
            states.append(state)
        return torch.cat(states[-self.concat_num:],dim=1)

class Network_Cifar(nn.Module):
    def __init__(self, initial_channel = 16, computing_node_num = 4, concat_num = 4, cell_num = 14, 
                 primitives = PRIMITIVES, affine = False, baseline = 0.02, auxiliary = False, num_classes = 10, 
                 channel_shuffle = False):
        super(Network_Cifar, self).__init__()
        self.cell_num = cell_num
        self.computing_node_num = computing_node_num
        self.concat_num = concat_num
        self.initial_channel = initial_channel
        
        self.stem_ops = nn.Sequential(
            nn.Conv2d(3, 3*initial_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channel*3, affine=True))

        self.cells = nn.ModuleDict()
        self.reduction_list = [cell_num//3, 2*cell_num//3]
        self.features_location = [cell_num//3 - 1, 2*cell_num//3 -1, cell_num - 1]
        self.auxiliary = auxiliary
        self.channels_list = []
        
        self.channel_shuffle = channel_shuffle
        
        if channel_shuffle:
            self.k = 2
        else:
            self.k = 1
        
        c_now = initial_channel
        c_prev_0, c_prev_1 = 3 * c_now, 3 * c_now
        
        reduction_prev_0 = 0
        reduction_prev_1 = 0
        
        for cell_order in range(cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            if cell_order in self.reduction_list:
                reduction = True 
                c_now = c_now*2
            else:   
                reduction = False
                
            self.cells[cell_name] = Cell(c_now, reduction, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,                                         
                                         cell_order, primitives, computing_node_num, concat_num, affine, baseline, 
                                         channel_shuffle, self.k)
            
            if cell_order in self.features_location:
                self.channels_list.append(c_prev_0)
            
            c_prev_0, c_prev_1 = concat_num*c_now, c_prev_0
            
            reduction_prev_1 = reduction_prev_1
            reduction_prev_0 = reduction

            if cell_order == 2*self.cell_num//3:
                c_to_auxiliary = c_prev_0

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev_0,num_classes, bias = True)
        
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCifar(c_to_auxiliary, num_classes) 

    def forward(self, inputs, architecture):
        features = []
        input_prev_0 = self.stem_ops(inputs)
        input_prev_1 = input_prev_0
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            state_now = self.cells[cell_name](input_prev_0, input_prev_1, architecture)
            input_prev_1, input_prev_0 = input_prev_0, state_now
            
            if cell_order in self.features_location:
                features.append(state_now)
            if cell_order == 2*self.cell_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(state_now)

        state_now = self.global_pooling(state_now)
        state_now = state_now.view(state_now.size()[0], -1)
        logits = self.classifier(state_now)
        
        if self.auxiliary and self.training:
            return logits, logits_aux, features
        return logits, logits, features
    

class Network_ImageNet(nn.Module):
    def __init__(self, initial_channel = 24, computing_node_num = 4, concat_num = 4, cell_num = 14,
                 primitives = PRIMITIVES, affine = False, baseline = 0.01, auxiliary = False, num_classes = 100):
        super(Network_ImageNet, self).__init__()
        self.cell_num = cell_num
        self.computing_node_num = computing_node_num
        self.concat_num = concat_num
        self.initial_channel = initial_channel

        self.cells = nn.ModuleDict()
        self.reduction_list = [cell_num//3, 2*cell_num//3]
        self.features_location = [cell_num//3 - 1, 2*cell_num//3 -1, cell_num - 1]
        self.auxiliary = auxiliary
        self.channels_list = []
        
        self.stem_0 = nn.Sequential(
          nn.Conv2d(3, initial_channel // 2, kernel_size = 3, stride = 2, padding = 1, bias = False),
          nn.BatchNorm2d(initial_channel // 2),
          nn.ReLU(inplace = True),
          nn.Conv2d(initial_channel // 2, initial_channel, 3, stride = 2, padding = 1, bias = False),
          nn.BatchNorm2d(initial_channel),
        )

        self.stem_1 = nn.Sequential(
          nn.ReLU(inplace = True),
          nn.Conv2d(initial_channel, initial_channel, 3, stride = 2, padding = 1, bias = False),
          nn.BatchNorm2d(initial_channel),
        )
        
        c_now = initial_channel
        c_prev_0, c_prev_1 = c_now, c_now
        
        reduction_prev_0 = True
        reduction_prev_1 = False
        
        for cell_order in range(cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            if cell_order in self.reduction_list:
                reduction = True 
                c_now = c_now*2
            else:   
                reduction = False  
            self.cells[cell_name] = Cell(c_now, reduction, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,                                         
                                         cell_order, primitives, computing_node_num, concat_num, affine, baseline)
            
            if cell_order in self.features_location:
                self.channels_list.append(c_prev_0)
            
            c_prev_0, c_prev_1 = concat_num*c_now, c_prev_0
            
            reduction_prev_1 = reduction_prev_0
            reduction_prev_0 = reduction

            if cell_order == 2*self.cell_num//3:
                c_to_auxiliary = c_prev_0

        self.global_pooling = nn.AvgPool2d(7)
        
        self.classifier = nn.Linear(c_prev_0, num_classes, bias = True)
        
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(c_to_auxiliary, num_classes) 

    def forward(self, inputs, architecture):
        features = []
        input_prev_1 = self.stem_0(inputs)
        input_prev_0 = self.stem_1(input_prev_1)
        
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            state_now = self.cells[cell_name](input_prev_0, input_prev_1, architecture)
            input_prev_1, input_prev_0 = input_prev_0, state_now
            
            if cell_order in self.features_location:
                features.append(state_now)
            if cell_order == 2*self.cell_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(state_now)

        state_now = self.global_pooling(state_now)
        state_now = state_now.view(state_now.size(0), -1)
        logits = self.classifier(state_now)
        
        if self.auxiliary and self.training:
            return logits, logits_aux, features
        return logits, logits, features


class Architecture(nn.Module):
    def __init__(self, cell_num = 14, computing_node_num = 4, primitives = PRIMITIVES,
                 option = 'relu', baseline = 0.01):
        super(Architecture, self).__init__()
        self.baseline = baseline
        self.cell_num = cell_num
        self.computing_node_num = computing_node_num
        self.para = nn.ParameterDict()
        self.option = option
        self.weights_dict = {}
        self.primitives = {}
        self.indicators = {}
        self.prune_buffer = []        
        
        for cell_order in range(cell_num):
            for computing_node_order in range(computing_node_num):
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                dimension = (computing_node_order + 2)*len(primitives)
                self.primitives[node_name] = []
                self.indicators[node_name] = []
                for input_order in range(computing_node_order + 2):
                    for _, op_name in enumerate(primitives):
                        self.indicators[node_name].append(input_order)
                        self.primitives[node_name].append(op_name)
                self.para[node_name] = nn.Parameter(2*torch.ones(dimension), requires_grad = True)
                # append initial weight to weights
                w = self.activating(self.para[node_name].data.detach().cpu())
                self.weights_dict[node_name] = w
        
    def activating(self, para, forward = False, c_point = 2.0, k = 0.1):
        # if forward:
        #     weights = F.relu(para)
        #     index = weights > c_point
        #     weights[index] = c_point + k*(weights[index] - c_point)
        #     weights = weights/sum(weights)
        # else:
        weights = F.relu(para)
        weights = weights/sum(weights)     
        return weights
    
    def calculate_sparsity(self, weights):
        return sum(torch.log(1.0 + 50*weights))
    
    def get_sparsity(self, grad = False):
        sparsity = 0.0
        for cell_order in range(self.cell_num):
            for intermediate_node_order in range(self.computing_node_num ):
                node_name = 'cell_{},node_{}'.format(cell_order, intermediate_node_order + 2)
                node_weights = self.get_weights(node_name)
                sparsity = sparsity + self.calculate_sparsity(node_weights)
        if grad:
            return sparsity
        else:
            return sparsity.data.detach().cpu()
        
    def get_weights(self, node_name, forward = False):
        para = self.para[node_name]
        weights = self.activating(para, forward = forward)
        return weights

    def prune(self):
        pruned_operators = []
        pruned_num = 0
        for cell_order in range(self.cell_num):
            for computing_node_order in range(self.computing_node_num):
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                para = self.para[node_name].data.detach().cpu()
                weights = self.activating(para)
                for order, weight in enumerate(weights):
                    if weight < self.baseline and self.para[node_name][order] > -300:                        
                        pruned_num = pruned_num + 1
                        with torch.no_grad():
                            self.para[node_name][order] = -1000.0 # delate architecture parameter
                        pruned_operators.append('cell_{},node_{},op:{}'.format(cell_order, computing_node_order, self.primitives[node_name][order]))
        self.prune_buffer.append(pruned_operators)
        
        return pruned_num, pruned_operators
        
    def append_weights(self):
        for cell_order in range(self.cell_num):
            for computing_node_order in range(self.computing_node_num):
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                para = self.para[node_name].data.detach().cpu()
                w = self.activating(para)
                self.weights_dict[node_name] = torch.vstack([self.weights_dict[node_name], w])
                
    def get_colors(self, primitives):
        primitives_colors = []
        for _,primitive in enumerate(primitives):
            if primitive == 'sep_conv_3':
                color = 'royalblue'
            elif primitive == 'sep_conv_5':
                color = 'blue'
            elif primitive == 'dil_conv_3':
                color = 'olive'
            elif primitive == 'dil_conv_5':
                color = 'darkgreen'
            elif primitive == 'max_pool_3':
                color = 'peru'
            elif primitive == 'avg_pool_3':
                color = 'orange'
            elif primitive == 'skip_connect':
                color = 'red'
            elif primitive == 'none':
                color = 'grey'
            primitives_colors.append(color)
        return primitives_colors
    
    def plot_weights(self, save_path = None, legend = False):
        for cell_order in range(self.cell_num):
            for computing_node_order in range(self.computing_node_num):
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                weights = self.weights_dict[node_name]
                primitives = self.primitives[node_name]

                # get primitives' colors
                colors = self.get_colors(primitives)
                
                x = np.linspace(0, len(weights), len(weights), dtype=int)                 
                # plt.figure(figsize=(10.0, 12.0))             
                for i in range(len(primitives)):
                    plt.plot(x, weights[:, i], color = colors[i], linewidth = 1.2)
                plt.xticks(fontsize = 5, fontweight = 'bold')
                plt.xlim(0, None)       
                
                if legend:            
                    # plt.ylabel('weights', fontsize=18)
                    plt.yticks(np.linspace(0, 1.0, 11), fontsize = 5, fontweight = 'bold')
                else:
                    plt.yticks([])
                plt.title(node_name, fontdict = {'family': 'Times New Roman', 'size':8, 'fontweight':'bold'})               
    
                if save_path is not None:
                    plt.savefig(os.path.join(save_path, node_name + '.jpg'))
                    plt.close()
                    
    def get_genotypes(self):
        self.primitives_finally = {}
        self.weights_finally = {}
        self.indicators_finally = {}
    
        for cell_order in range(self.cell_num):
            for computing_node_order in range(self.computing_node_num):
                weights_ = []
                primitives_ = []
                indicators_ = []
                node_name = 'cell_{},node_{}'.format(cell_order, computing_node_order + 2)
                weights = list(self.get_weights(node_name).data.detach().cpu().numpy())
                primitives = self.primitives[node_name]
                indicators = self.indicators[node_name]
                for i in range(len(primitives)):
                    if weights[i] < self.baseline:
                        pass
                    else:
                        weights_.append(weights[i])
                        primitives_.append(primitives[i])
                        indicators_.append(indicators[i])
                self.weights_finally[node_name] = weights_
                self.primitives_finally[node_name] = primitives_
                self.indicators_finally[node_name] = indicators_
                
        genotypes = {'weights_finally' : self.weights_finally, 'primitives_finally' : self.primitives_finally,
                     'indicators_finally' : self.indicators_finally, 'cell_num' : self.cell_num,
                     'computing_node_num' : self.computing_node_num}
        
        return genotypes    
            
class AuxiliaryHeadCifar(nn.Module):
    # 假设inputs经过Network的输出后变成8×8大小    
    def __init__(self,c_inputs,num_classes):        
        super(AuxiliaryHeadCifar,self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AvgPool2d(5,stride=2,padding=0,count_include_pad=False),
            nn.Conv2d(c_inputs,128,kernel_size=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128,768,2,bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=False)
            )
        self.classifier  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768,num_classes,bias=True),
            )
    def forward(self,inputs):
        inputs = self.features(inputs)
        inputs = inputs.view(inputs.size(0),-1)
        outputs = self.classifier(inputs)
        return outputs
    
class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace = True),
      nn.AvgPool2d(5, stride = 2, padding = 0, count_include_pad = False),
      nn.Conv2d(C, 128, 1, bias = False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace = True),
      nn.Conv2d(128, 768, 2, bias = False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace = True)
    )
    
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class Distillartor(nn.Module):
    def __init__(self, student_channel_list = [64, 128, 258], teacher_channel_list = [192, 384, 768]):
        super(Distillartor, self).__init__()
        self.s_layer_0 = nn.Sequential(nn.Conv2d(student_channel_list[0], teacher_channel_list[0], kernel_size = 1, padding = 0),
                                       nn.BatchNorm2d(teacher_channel_list[0], affine = False))
        
        self.s_layer_1 = nn.Sequential(nn.Conv2d(student_channel_list[1], teacher_channel_list[1], kernel_size = 1, padding = 0),
                                       nn.BatchNorm2d(teacher_channel_list[1], affine = False))
        
        self.s_layer_2 = nn.Sequential(nn.Conv2d(student_channel_list[2], teacher_channel_list[2], kernel_size = 1, padding = 0),
                                       nn.BatchNorm2d(teacher_channel_list[2], affine = False))
        
        # self.bn = nn.BatchNorm2d(teacher_channel, affine = False)
        self.t_layer_0 = nn.Sequential(nn.BatchNorm2d(teacher_channel_list[0], affine = False))

        self.t_layer_1 = nn.Sequential(nn.BatchNorm2d(teacher_channel_list[1], affine = False))

        self.t_layer_2 = nn.Sequential(nn.BatchNorm2d(teacher_channel_list[2], affine = False))       
        
    def forward(self, student_features, teacher_features, ratio = [0.2, 0.3, 0.5]):
        s_f_0 = self.s_layer_0(student_features[0])
        s_f_1 = self.s_layer_1(student_features[1])
        s_f_2 = self.s_layer_2(student_features[2]) 
        
        t_f_0 = self.t_layer_0(teacher_features[0])
        t_f_1 = self.t_layer_1(teacher_features[1])
        t_f_2 = self.t_layer_2(teacher_features[2])   
        
        loss = ratio[0]*F.mse_loss(s_f_0, t_f_0) + ratio[1]*F.mse_loss(s_f_1, t_f_1) + ratio[2]*F.mse_loss(s_f_2, t_f_2)
        return loss

    
class Pruner(nn.Module):
    def __init__(self, gamma_0 = 0.5, gamma_max = 1.0, varphi = 1.5, k = 0.9, n_max = 8, tqu_max = 5, tau = 0.90, miu_0 = 1e-2, rou = 0.8, xi = 0.8):
        super(Pruner, self).__init__()
        
        self.gamma_0 = gamma_0
        self.varphi = varphi
        self.gamma_max = gamma_max
        self.k = k
        self.gamma = gamma_0
        self.n_max = n_max
        self.tqu_max = tqu_max
        self.tqu = 0
        self.prune_step = 0
        self.gamma_list = []
        self.flops_list = []
        self.pruned_num = 0
        self.expected_pruned_num  = 0
        self.tau = tau
        self.miu_0 = miu_0
        self.miu = miu_0
        self.rou = rou
        self.xi = xi
        self.flag = False
        self.sparsity_list = []
        self.miu_list = []
        
        
    def update_sparsity(self, architecture):
        self.sparsity_list.append(architecture.get_sparsity(grad = False))
        self.gamma_list.append(0.0)
        
    def prune(self, architecture, network):
        self.expected_pruned_num = self.expected_pruned_num + self.n_max
        pareto_genotype = False
        p_num, delate = architecture.prune()
        self.pruned_num = self.tau*self.pruned_num + (1 - self.tau)*p_num
        if self.pruned_num < self.n_max:
            if self.gamma < self.gamma_max:
                self.gamma = self.varphi*self.gamma 
                self.tqu = 0
            elif self.gamma >= self.gamma_max:
                self.gamma = self.varphi*self.gamma
                self.tqu = self.tqu + 1
                if self.tqu >= self.tqu_max:
                    pareto_genotype = True 
                    self.miu = self.rou*self.miu
                    self.tqu = 0
                    self.n_max = self.xi * self.n_max           
        else:
            self.gamma = self.k * self.gamma
            self.tqu = 0
            
        self.gamma_list.append(self.gamma)
        self.sparsity_list.append(architecture.get_sparsity(grad = False))
        self.miu_list.append(self.miu)
        return pareto_genotype, delate
    
    def plot_fig(self, save_path = None):
        ax1 = plt.subplot(1, 2, 1)
        # plt.plot(self.gamma_list, c = 'red', linewidth = 1.5)
        plt.semilogy(self.gamma_list, c = 'red', linewidth = 1.5)
        plt.ylabel('gamma')
        plt.xlabel('prune_step')
        ax2 = plt.subplot(1, 2, 2)
        plt.plot(self.sparsity_list, color = 'blue', linewidth = 1.5)
        plt.ylabel('FLOPs')
        plt.xlabel('prune_step')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'FLOPs.jpg'))
            plt.close()