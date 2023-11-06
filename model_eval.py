import torch
import torch.nn as nn
import numpy as np
from operations_eval import *
from utils_ import drop_path, drop_path_identity


class Normal_Computing_Node(nn.Module):
    def __init__(self, c_now, node_id, affine = True, primitives = None, indicators = None):
        super(Normal_Computing_Node, self).__init__()
        self.c_now = c_now
        self.node_id = node_id 
        self.primitives = primitives
        self.indicators = indicators
        self.ops = nn.ModuleList()
        self.drop_prob = 0.0
        
        for primitive, input_order in zip(primitives, indicators):
            op = OPS[primitive](c_now, 1, affine)
            self.ops.append(op)
    
    def forward(self, states):
        state = 0.0
        for op, input_order in zip(self.ops, self.indicators):
            h = op(states[input_order])
            if self.training and self.drop_prob > 0 :
                if not isinstance(op, Identity):
                    h = drop_path(h, self.drop_prob)
            state = state + h
        return state
    
    def update_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
        
class Reduction_Computing_Node(nn.Module):
    def __init__(self, c_now, node_id, affine = True, primitives = None, indicators = None):
        super(Reduction_Computing_Node, self).__init__()
        self.c_now = c_now
        self.node_id = node_id 
        self.primitives = primitives
        self.indicators = indicators
        self.ops = nn.ModuleList()
        self.drop_prob = 0.0
        
        for primitive, input_order in zip(primitives, indicators):
            stride = 2 if input_order < 2 else 1
            op = OPS[primitive](c_now, stride, affine)
            self.ops.append(op)
    
    def forward(self, states):
        state = 0.0
        for op, input_order in zip(self.ops, self.indicators):
            h = op(states[input_order])
            if self.training and self.drop_prob > 0 :
                if not isinstance(op, Identity):
                    h = drop_path(h, self.drop_prob)
            state = state + h
        return state
    
    def update_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
        
class Cell(nn.Module):
    def __init__(self, c_now, reduction_now, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,
                 cell_id, computing_node_num = 4, concat_num = 4, affine = True, genotype = None):
        super(Cell, self).__init__()
        self.c_now = c_now
        self.c_prev_0 = c_prev_0
        self.c_prev_1 = c_prev_1
        self.cell_id = cell_id
        self.reduction = reduction_now
        self.reduction_prev_0 = reduction_prev_0
        self.reduction_prev_1 = reduction_prev_1  
        self.concat_num = concat_num
        self.computing_node_num = computing_node_num
        self.preprocess_0 = ReLUConvBN(c_prev_0, c_now, kernel_size = 1, stride = 1, padding = 0, affine = affine)
        if reduction_prev_0:            
            self.preprocess_1 = ReduceOperator(c_prev_1, c_now, affine = affine)
        else:
            self.preprocess_1 = ReLUConvBN(c_prev_1, c_now, kernel_size = 1, stride = 1, padding = 0, affine = affine)
        self.computing_nodes = nn.ModuleList()
        
        for computing_node_order in range(computing_node_num):
            node_name = 'cell_{},node_{}'.format(cell_id, computing_node_order + 2)
            
            node_primitives = genotype['primitives_finally'][node_name]
            node_indicators = genotype['indicators_finally'][node_name]
            
            if reduction_now:
                computing_node = Reduction_Computing_Node(c_now, computing_node_order + 2, affine, node_primitives, node_indicators)
            else:
                computing_node = Normal_Computing_Node(c_now, computing_node_order + 2, affine, node_primitives, node_indicators)
            self.computing_nodes.append(computing_node)
        
    def forward(self, input_prev_0, input_prev_1):
        state_prev_0 = self.preprocess_0(input_prev_0)
        state_prev_1 = self.preprocess_1(input_prev_1)
        states = [state_prev_0, state_prev_1]
        for computing_node_order in range(self.computing_node_num):
            state = self.computing_nodes[computing_node_order](states)
            states.append(state)
        return torch.cat(states[-self.concat_num:],dim=1)
    
    def update_drop_prob(self, drop_prob):
        for computing_node_order in range(self.computing_node_num):
            self.computing_nodes[computing_node_order].update_drop_prob(drop_prob)
            self.drop_prob = drop_prob
            
class Network_Cifar(nn.Module):
    def __init__(self, initial_channel = 16, auxiliary = False, num_classes = 10, genotypes = None):
        super(Network_Cifar, self).__init__()
        self.genotypes = genotypes
        self.cell_num = genotypes['cell_num']        
        self.reduction_list = [self.cell_num//3, 2*self.cell_num//3]       
        self.computing_node_num = genotypes['computing_node_num']
        self.stem_ops = nn.Sequential(nn.Conv2d(3, 3*initial_channel, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(initial_channel*3, affine=True))
        self.cells = nn.ModuleDict()
        self.auxiliary = auxiliary
        self.concat_num = 4
              
        c_now = initial_channel
        c_prev_0, c_prev_1 = 3 * c_now, 3 * c_now        
        reduction_prev_0 = 0
        reduction_prev_1 = 0   
        
        primitives = genotypes['primitives_finally']
        indicators = genotypes['indicators_finally']
        computing_node_num = genotypes['computing_node_num']
        
        def decide_wheather_to_add(cell_order, total_cell_num = 14):
            add = False
            if cell_order < total_cell_num - 1:
                for computing_node_order in range(computing_node_num):
                    next_node_name = 'cell_{},node_{}'.format(cell_order + 1, computing_node_order + 2)
                    next_node_indicators = indicators[next_node_name]
                    if 0 in next_node_indicators:
                        add = True
            if cell_order < total_cell_num - 2:
                for computing_node_order in range(computing_node_num):
                    next_node_name = 'cell_{},node_{}'.format(cell_order + 2, computing_node_order + 2)
                    next_node_indicators = indicators[next_node_name]
                    if 1 in next_node_indicators:
                        add = True
            if cell_order == total_cell_num - 1:
                add = True
            return add
        
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            
            add = decide_wheather_to_add(cell_order, self.cell_num)
            
            if add:     
                if cell_order in self.reduction_list:
                    reduction = True 
                    c_now = c_now*2
                else:   
                    reduction = False
                
                self.cells[cell_name] = Cell(c_now, reduction, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,                                         
                                             cell_order, self.computing_node_num, 4, True, genotypes)
            else:
                self.cells[cell_name] = None
                 
            
            c_prev_0, c_prev_1 = self.concat_num*c_now, c_prev_0            
            reduction_prev_1 = reduction_prev_1
            reduction_prev_0 = reduction
            
            if cell_order == 2*self.cell_num//3:
                c_to_auxiliary = c_prev_0
        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadCifar(c_to_auxiliary, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev_0, num_classes, bias = True)
        
    def forward(self, inputs, obtain_features = False, features_location = None):
        features = []
        if obtain_features and features_location is not None:
            self.features_location = features_location

        state_0 = self.stem_ops(inputs)
        state_1 = state_0
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            if self.cells[cell_name] is not None:         
                state_now = self.cells[cell_name](state_0, state_1)
                state_0, state_1 = state_now, state_0
            else:
                state_now = None
            
            if cell_order == 2*self.cell_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(state_now)
            
            if obtain_features:       
                if cell_order in self.features_location:
                    features.append(state_now)
        
        state_now = self.global_pooling(state_now)
        state_now = state_now.view(state_now.size(0),-1)
        logits = self.classifier(state_now)

        if obtain_features:     
            return logits, features
        elif self.auxiliary and self.training:            
            return logits, logits_aux
        else:
            return logits, None
        
        
    def update_drop_prob(self, drop_prob):
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            if self.cells[cell_name] is not None:
                self.cells[cell_name].update_drop_prob(drop_prob)

class AuxiliaryHeadCifar(nn.Module):
    # 假设inputs经过Network的输出后变成8×8大小    
    def __init__(self, c_inputs, num_classes):        
        super(AuxiliaryHeadCifar, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace = False),
            nn.AvgPool2d(5, stride=2, padding = 0, count_include_pad = False),
            nn.Conv2d(c_inputs, 128, kernel_size = 1,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            nn.Conv2d(128, 768, 2, bias = False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace = False))
        self.classifier = nn.Linear(768, num_classes, bias = True)
            
    def forward(self,inputs):
        inputs = self.features(inputs)
        inputs = inputs.view(inputs.size(0),-1)
        outputs = self.classifier(inputs)
        return outputs
        
class Network_Imagenet(nn.Module):
    def __init__(self, initial_channel = 16, auxiliary = False, num_classes = 10, genotypes = None):
        super(Network_Imagenet, self).__init__()
        self.cells = nn.ModuleDict()
        self.cell_num = genotypes['cell_num']
        self.auxiliary = auxiliary
        self.concat_num = 4   
        self.genotype = genotypes
        self.reduction_list = [self.cell_num//3, 2*self.cell_num//3]
        self.features_location = [self.cell_num//3 - 1, 2*self.cell_num//3 -1, self.cell_num - 1]
        self.computing_node_num = genotypes['computing_node_num']
        
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
        

        reduction = 0
        reduction_prev_0 = 1
        reduction_prev_1 = 0     
        c_now = initial_channel
        
        c_prev_0, c_prev_1 = c_now, c_now
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            if cell_order in self.reduction_list:
                reduction = True 
                c_now = c_now*2
            else:   
                reduction = False  
            self.cells[cell_name] = Cell(c_now, reduction, c_prev_0, reduction_prev_0, c_prev_1, reduction_prev_1,                                         
                                         cell_order, self.computing_node_num, 4, True, genotypes)

            c_prev_0, c_prev_1 = self.concat_num*c_now, c_prev_0            
            reduction_prev_1 = reduction_prev_1
            reduction_prev_0 = reduction
            
            if cell_order == 2*self.cell_num//3:
                c_to_auxiliary = c_prev_0

        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(c_to_auxiliary, num_classes)
            
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(c_prev_0,num_classes,bias = True)
        
    def forward(self, inputs):
        features = []
        state_1 = self.stem_0(inputs)
        state_0 = self.stem_1(state_1)
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            state_now = self.cells[cell_name](state_0, state_1)
            state_0,state_1 = state_now, state_0
            
            if cell_order in [self.cell_num//3, 2*self.cell_num//3]:
                features.append(state_now)
            if cell_order == self.cell_num - 1:
                features.append(state_now)
            if cell_order == 2*self.cell_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(state_now)      
        state_now = self.global_pooling(state_now)
        state_now = state_now.view(state_now.size(0),-1)
        logits = self.classifier(state_now)
        
        if self.auxiliary and self.training:            
            return logits,logits_aux,features
        else:
            return logits,logits,features
        
    def update_drop_prob(self, drop_prob):
        for cell_order in range(self.cell_num):
            cell_name = 'cell_{}'.format(cell_order)
            self.cells[cell_name].update_drop_prob(drop_prob)
        
class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x
    
        
        