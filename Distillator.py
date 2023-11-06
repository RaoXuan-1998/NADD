import torch
import torch.nn as nn
import torch.nn.fufunctional as F

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T = 4):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim = 1),
						F.softmax(out_t/self.T, dim = 1),
						reduction = 'batchmean') * self.T * self.T

		return loss

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p = 2, beta = 1000):
		super(AT, self).__init__()
		self.p = p
		self.beta = beta
        
	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return self.beta*loss

	def attention_map(self, fm, eps = 1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim = 1, keepdim = True)
		norm = torch.norm(am, dim = (2,3), keepdim = True)
		am = torch.div(am, norm + eps)
        
		return am
    
class FT(nn.Module):
	'''
	araphrasing Complex Network: Network Compression via Factor Transfer
	http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf
	'''
	def __init__(self, beta = 100):
		super(FT, self).__init__()
		self.beta = beta

	def forward(self, factor_s, factor_t):
		loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))

		return self.beta*loss

	def normalize(self, factor):
        
		norm_factor = F.normalize(factor.view(factor.size(0),-1))

		return norm_factor
    
class Distillartor(nn.Module):
    def __init__(self, KD_Type = 'FT'):
        super(Distillartor, self).__init__()
        self.KD_Type = KD_Type
        if KD_Type == 'ST':
            self.ST = SoftTarget()
        
        elif KD_Type == 'AT':
            self.AT = AT()
        
        elif KD_Type == 'FT':
            self.FT = FT()

        
    def forward(self,
                student_features = None, teacher_features = None,
                student_logits = None, teacher_logits = None):
        
        if self.KD_Type == 'ST':
            KD_loss = self.ST(student_logits, teacher_logits)
        
        elif self.KD_Type == 'AT':
            KD_loss = self.AT(student_features[0], teacher_features[0]) + \
                self.AT(student_features[1], teacher_features[1]) + \
                self.AT(student_features[2], teacher_features[2])
            KD_loss = KD_loss/3
            
        
        elif self.KD_Type == 'FT':
            KD_loss = self.FT(student_features[2], teacher_features[2])
        
        return KD_loss

    
    