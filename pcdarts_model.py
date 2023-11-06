# In[0]
import torch
import torch.nn as nn
from operations_eval import *
from genotypes import *
from utils_ import drop_path

# In[1]
class reduction_cell(nn.Module):
    def __init__(self,c_now,reduction_now,
                 c_prev_0,reduction_prev_0,
                 c_prev_1,reduction_prev_1,
                 intermediate_nodes_num=4,
                 primitives=PRIMITIVES,
                 affine=True,
                 concat_num=4):
        super(reduction_cell,self).__init__()
        self.preprocess_0 = ReLUConvBN(c_prev_0,c_now,kernel_size=1,stride=1,
                                       padding=0,affine=affine)
        self.concat_num = concat_num
        if reduction_prev_0:            
            self.preprocess_1 = ReduceOperator(c_prev_1,c_now,affine=affine)
        else:
            self.preprocess_1 = ReLUConvBN(c_prev_1,c_now,kernel_size=1,stride=1,
                                           padding=0,affine=affine)
        self.edges = nn.ModuleDict()
        self.edges['edge_(0,2)'] = OPS['sep_conv_5'](c_now,2,affine)
        self.edges['edge_(1,2)'] = OPS['max_pool_3'](c_now,2,affine)
        self.edges['edge_(0,3)'] = OPS['sep_conv_5'](c_now,2,affine)
        self.edges['edge_(2,3)'] = OPS['sep_conv_5'](c_now,1,affine)
        self.edges['edge_(1,4)'] = OPS['sep_conv_3'](c_now,2,affine)
        self.edges['edge_(3,4)'] = OPS['sep_conv_3'](c_now,1,affine)
        self.edges['edge_(0,5)'] = OPS['sep_conv_3'](c_now,2,affine)
        self.edges['edge_(2,5)'] = OPS['sep_conv_3'](c_now,1,affine)
    def forward(self,input_0,input_1,drop_prob):
        state_0 = self.preprocess_0(input_0)
        state_1 = self.preprocess_1(input_1)
        if self.training and drop_prob > 0:
            state_2 = drop_path(self.edges['edge_(0,2)'](state_0),drop_prob) + drop_path(self.edges['edge_(1,2)'](state_1),drop_prob)
            state_3 = drop_path(self.edges['edge_(0,3)'](state_0),drop_prob) + drop_path(self.edges['edge_(2,3)'](state_2),drop_prob)
            state_4 = drop_path(self.edges['edge_(1,4)'](state_1),drop_prob) + drop_path(self.edges['edge_(3,4)'](state_3),drop_prob)
            state_5 = drop_path(self.edges['edge_(0,5)'](state_0),drop_prob) + drop_path(self.edges['edge_(2,5)'](state_2),drop_prob)
        else:
            state_2 = self.edges['edge_(0,2)'](state_0) + self.edges['edge_(1,2)'](state_1)
            state_3 = self.edges['edge_(0,3)'](state_0) + self.edges['edge_(2,3)'](state_2)
            state_4 = self.edges['edge_(1,4)'](state_1) + self.edges['edge_(3,4)'](state_3)
            state_5 = self.edges['edge_(0,5)'](state_0) + self.edges['edge_(2,5)'](state_2)
        state = [state_2,state_3,state_4,state_5]
        return torch.cat(state[-self.concat_num:],dim=1)
    
class normal_cell(nn.Module):
    def __init__(self,c_now,reduction_now,
                 c_prev_0,reduction_prev_0,
                 c_prev_1,reduction_prev_1,
                 intermediate_nodes_num=4,
                 primitives=PRIMITIVES,
                 affine=True,
                 concat_num=4):
        super(normal_cell,self).__init__()
        
        self.concat_num = concat_num
        self.preprocess_0 = ReLUConvBN(c_prev_0,c_now,kernel_size=1,stride=1,
                                       padding=0,affine=affine)
        if reduction_prev_0:            
            self.preprocess_1 = ReduceOperator(c_prev_1,c_now,affine=affine)
        else:
            self.preprocess_1 = ReLUConvBN(c_prev_1,c_now,kernel_size=1,stride=1,
                                           padding=0,affine=affine)
        self.edges = nn.ModuleDict()
        self.edges['edge_(0,2)'] = OPS['sep_conv_3'](c_now,1,affine)
        self.edges['edge_(1,2)'] = OPS['skip_connect'](c_now,1,affine)
        self.edges['edge_(0,3)'] = OPS['dil_conv_3'](c_now,1,affine)
        self.edges['edge_(1,3)'] = OPS['sep_conv_3'](c_now,1,affine)
        self.edges['edge_(0,4)'] = OPS['sep_conv_3'](c_now,1,affine)
        self.edges['edge_(1,4)'] = OPS['sep_conv_5'](c_now,1,affine)
        self.edges['edge_(0,5)'] = OPS['dil_conv_3'](c_now,1,affine)
        self.edges['edge_(1,5)'] = OPS['avg_pool_3'](c_now,1,affine)

    def forward(self,input_0,input_1, drop_prob = 0.3):
        state_0 = self.preprocess_0(input_0)
        state_1 = self.preprocess_1(input_1)
        if self.training and drop_prob > 0:
            state_2 = drop_path(self.edges['edge_(0,2)'](state_0),drop_prob) + self.edges['edge_(1,2)'](state_1)
            state_3 = drop_path(self.edges['edge_(0,3)'](state_0),drop_prob) + drop_path(self.edges['edge_(1,3)'](state_1),drop_prob)
            state_4 = drop_path(self.edges['edge_(0,4)'](state_0),drop_prob) + drop_path(self.edges['edge_(1,4)'](state_1),drop_prob)
            state_5 = drop_path(self.edges['edge_(0,5)'](state_0),drop_prob) + drop_path(self.edges['edge_(1,5)'](state_1),drop_prob)
        else:
            state_2 = self.edges['edge_(0,2)'](state_0) + self.edges['edge_(1,2)'](state_1)
            state_3 = self.edges['edge_(0,3)'](state_0) + self.edges['edge_(1,3)'](state_1)
            state_4 = self.edges['edge_(0,4)'](state_0) + self.edges['edge_(1,4)'](state_1)
            state_5 = self.edges['edge_(0,5)'](state_0) + self.edges['edge_(1,5)'](state_1)
        state = [state_2,state_3,state_4,state_5]
        return torch.cat(state[-self.concat_num:],dim=1)

class PC_DARTS(nn.Module):
    def __init__(self,
                 cells_num=14, 
                 intermediate_nodes_num=4,
                 concat_num=4,
                 num_classes=10,
                 c_start=16,
                 stem_multiplier=3,
                 primitives=PRIMITIVES,
                 auxiliary=True,
                 affine=True):
        super(PC_DARTS,self).__init__()
        self.stem_ops = nn.Sequential(
            nn.Conv2d(3,stem_multiplier*c_start,3,padding=1,bias=False),
            nn.BatchNorm2d(c_start*stem_multiplier))
        self.cells = nn.ModuleDict()
        self.cells_num = cells_num
        self.primitives = primitives
        self.auxiliary = auxiliary
        self.concat_num = concat_num
        cell_reduction_now = 0
        cell_prev_0_reduction = 0
        cell_prev_1_reduction = 0     
        c_now = c_start
        c_prev_0,c_prev_1, = stem_multiplier * c_now,stem_multiplier * c_now
        for cell_order in range(cells_num):
            cell_name = 'cell_{}'.format(cell_order)
            if cell_order in [cells_num//3,2*cells_num//3]:
                cell_reduction_now = True
                c_now = c_now*2
                self.cells[cell_name] = reduction_cell(
                    c_now,cell_reduction_now,
                    c_prev_0,cell_prev_0_reduction,
                    c_prev_1,cell_prev_1_reduction,
                    intermediate_nodes_num=4,
                    primitives=self.primitives,
                    affine=affine,
                    concat_num=self.concat_num)
            else:
                cell_reduction_now = False
                self.cells[cell_name] = normal_cell(
                    c_now,cell_reduction_now,
                    c_prev_0,cell_prev_0_reduction,
                    c_prev_1,cell_prev_1_reduction,
                    intermediate_nodes_num=4,
                    primitives=self.primitives,
                    affine=affine,
                    concat_num=self.concat_num)

            c_prev_0,c_prev_1 = self.concat_num*c_now,c_prev_0           
            cell_prev_0_reduction = cell_reduction_now
            cell_prev_1_reduction = cell_prev_0_reduction
            
            if cell_order == 2*self.cells_num//3:
                c_to_auxiliary = c_prev_0
        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadCifar(c_to_auxiliary, num_classes)
            
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev_0,num_classes,bias=True)

    def forward(self,inputs,drop_prob):
        features = []
        state_1 = self.stem_ops(inputs)
        state_0 = self.stem_ops(inputs)
        for cell_order in range(self.cells_num):
            cell_name = 'cell_{}'.format(cell_order)
            state_now = self.cells[cell_name](state_0,state_1,drop_prob)
            state_0,state_1 = state_now,state_0
            
            if cell_order in [self.cells_num//3 - 1, 2*self.cells_num//3 - 1, self.cells_num - 1]:
                features.append(state_now)
            if cell_order == 2*self.cells_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(state_now)
        
        state_now = self.global_pooling(state_now)
        state_now = state_now.view(state_now.size(0),-1)
        logits = self.classifier(state_now)
        
        if self.auxiliary and self.training:            
            return logits,logits_aux,features
        else:
            return logits, logits, features

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
        self.classifier = nn.Linear(768,num_classes,bias=True)
            
        
    def forward(self,inputs):
        inputs = self.features(inputs)
        inputs = inputs.view(inputs.size(0),-1)
        outputs = self.classifier(inputs)
        return outputs


