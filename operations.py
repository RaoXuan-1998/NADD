# In[0]
import torch
import torch.nn as nn

# In[1]
OPS = {'none':lambda c,stride,affine : Zeros(stride),
       'sep_conv_3':lambda c,stride,affine : SepConv(c,c,3,stride,1,affine=affine),
       'sep_conv_5':lambda c,stride,affine : SepConv(c,c,5,stride,2,affine=affine),
       'sep_conv_7':lambda c,stride,affine : SepConv(c,c,7,stride,3,affine=affine),
       'dil_conv_3':lambda c,stride,affine : DilConv(c,c,3,stride,2,2,affine=affine),
       'dil_conv_5':lambda c,stride,affine : DilConv(c,c,5,stride,4,2,affine=affine),
       'skip_connect':lambda c,stride,affine : Identity(c,c,affine) if stride==1 else ReduceOperator(c,c,affine),
       'conv_7x1_1x7':lambda c,stride,affine : Conv_7x1_1x7(c,c,stride,affine=affine),
       'avg_pool_3':lambda c,stride,affine : AvgPool(c,c,3,stride,1,affine=affine),
       'max_pool_3' : lambda c, stride, affine : MaxPool(c,c,3,stride,1,affine=affine)}

class SepConv(nn.Module):
    def __init__(self,c_in, c_out, kernel_size, stride,padding, affine=True):
        super(SepConv,self).__init__()      
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in,c_in,kernel_size=kernel_size,stride=stride,
                      padding=padding,groups=c_in,bias=False),
            nn.Conv2d(c_in,c_in,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(c_in,affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in,c_in,kernel_size=kernel_size,stride=1,
                      padding=padding,groups=c_in,bias=False),
            nn.Conv2d(c_in,c_out,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(c_out,affine=affine))
        
    def forward(self,x):
        return self.op(x)
    
    def flops(self):
        return self.flops

# In[3]
class Identity(nn.Module):
    def __init__(self,c_in,c_out,affine=True):
        super(Identity,self).__init__()
        self.flops = 0.
        
    def forward(self,x):
        return x
    
    def flops(self):
        return 0.
# In[4]
class Zeros(nn.Module):
    
    def __init__(self,stride):
        super(Zeros,self).__init__()
        self.stride = stride
        self.flops = 0.
    def forward(self,x):
        if self.stride == 1:
            x = x.mul(0.)
        else:
            x = x[:,:,::self.stride,::self.stride].mul(0.)
        return x
    
    def flops(self):
        return 0.

# In[4]
class ReduceOperator(nn.Module):
    
    def __init__(self,c_in,c_out,affine=True):
        super(ReduceOperator,self).__init__()
        assert c_out%2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(c_in,c_out//2,kernel_size=1,stride=2,
                               padding=0,bias=False)
        self.conv2 = nn.Conv2d(c_in,c_out//2,kernel_size=1,stride=2,
                               padding=0,bias=False)
        self.bn = nn.BatchNorm2d(c_out,affine=affine)
    def forward(self,x):
        x = self.relu(x)
        x = torch.cat([self.conv1(x),self.conv2(x[:,:,1:,1:])],dim=1)
        out = self.bn(x)
        return out

# In[5]
class ReLUConvBN(nn.Module):
    
    def __init__(self,c_in,c_out,kernel_size,stride,padding,affine=True):
        super(ReLUConvBN, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in,c_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(c_out,affine=affine)
            )
    def forward(self,x):
        x = self.ops(x)
        return x
# In[6]
class Conv_7x1_1x7(nn.Module):
    
    def __init__(self,c_in,c_out,stride,affine=True):
        super(Conv_7x1_1x7,self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in,c_in,(1,7),stride=(1,stride),padding=(0,3),bias=False),
            nn.Conv2d(c_in,c_out,(7,1),stride=(stride,1),padding=(3,0),bias=False),
            nn.BatchNorm2d(c_out,affine=affine)
            )
    def forward(self,x):
        return self.ops(x)

# In[7]
class DilConv(nn.Module):
    def __init__(self,c_in,c_out,kernel_size,stride,padding,dilation,affine=True):
        super(DilConv,self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in,c_in,kernel_size=kernel_size,stride=stride,
                      padding=padding,dilation=dilation,groups=c_in,bias=False),
            nn.Conv2d(c_in,c_out,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(c_out,affine=affine)
            )
    def forward(self,x):
        return self.ops(x)
# In[8]
class MaxPool(nn.Module):
    
    def __init__(self,c_in,c_out,kernel_size,stride,padding,affine=True):
        super(MaxPool,self).__init__()
        self.ops = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(c_out,affine=affine)
            )
    def forward(self,x):
        return self.ops(x)


class AvgPool(nn.Module):
    def __init__(self,c_in,c_out,kernel_size,stride,padding,affine=True):
        super(AvgPool,self).__init__()
        self.ops = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size,stride=stride,
                         padding=padding,count_include_pad=False),
            nn.BatchNorm2d(c_out,affine=affine)
            )
    def forward(self,x):
        return self.ops(x)
    def flops(self):
        return self.flops
            