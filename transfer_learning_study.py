import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import utils_
import torch.utils
import torchvision.transforms as transforms
import argparse
import torchvision.datasets as datasets
import logging
import torch.backends.cudnn as cudnn
import time
from model_eval import Network_Imagenet
from auto_augment import CIFAR10Policy
# 这个迁移学习是直接微调整个网络的权重

parser = argparse.ArgumentParser('Fine-tune ImageNet network on CIFAR-10/100')
parser.add_argument('--save', type = str, default = 'transger-learning-cifar100')
parser.add_argument('--data', type = str, default = '../../data')
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--steps', type = int, default = 50000)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--num_classes', type = int, default = 100)
parser.add_argument('--learning_rate', type = float, default = 1e-3)
parser.add_argument('--learning_rate_min', type = float , default = 0.0)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--cutout', action='store_false', default = True)
parser.add_argument('--cutout_length', type = int, default = 112)
parser.add_argument('--auto_augment', type = bool, default = False)
parser.add_argument('--pin_memory', type = bool, default = True)
parser.add_argument('--num_workers', type = int, default = 0)
parser.add_argument('--weight_decay', type = float, default = 5e-4)
parser.add_argument('--model_path', type = str, default = 'eval-imagenet-10-epoch_20,geno(4)/model_best.pth.tar')
parser.add_argument('--load_geno_path', type=str, default='search-imagenet-10/genotypes/epoch_20,geno.npy')
parser.add_argument('--channels',type=int,default = 48,help = 'c_start')
parser.add_argument('--auxiliary',type=bool,default = True)
args,unparsed = parser.parse_known_args()

utils_.create_exp_dir(args.save)
# utils_.create_exp_dir(args.save, scripts_to_save = glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


network_genotype = np.load(args.load_geno_path, allow_pickle = True).tolist()
model = Network_Imagenet(args.channels, args.auxiliary, 1000, network_genotype).cuda()
model_dict = torch.load(args.model_path, map_location = 'cuda:0')
model.load_state_dict(model_dict['state_dict'])
model.auxiliary = False
# model.update_drop_prob(0.3)

model.classifier = nn.Linear(768, args.num_classes).cuda()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(args.gpu)

logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)



train_transform, valid_transform = utils_._data_transforms_cifar10_224(args)

train_data = datasets.CIFAR100(root = args.data, train = True, download = True, transform = train_transform)
valid_data = datasets.CIFAR100(root = args.data, train = False, download = True, transform = valid_transform)


train_queue = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, pin_memory = args.pin_memory,
                                          num_workers = args.num_workers)  
valid_queue = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, pin_memory = args.pin_memory, 
                                          num_workers = args.num_workers)

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(args.steps),
                                                               eta_min = args.learning_rate_min)

total_step = 0
model.train()
top1 = utils_.AvgrageMeter()
top5 = utils_.AvgrageMeter()
while total_step < args.steps:
    for step, (inputs,targets) in enumerate(train_queue):
        
        n = inputs.size(0)
        
        total_step = total_step + 1 
        
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        total_step = total_step + 1
        
        logits = model(inputs)[0]
        loss = criterion(logits, targets)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        prec_1,prec_5 = utils_.accuracy(logits,targets,topk=(1, 5))
        top1.update(prec_1.data.item(), n)
        top5.update(prec_5.data.item(), n)    
        
        if total_step % 100 == 0 or total_step == args.steps - 1:
            logging.info('step:{} '.format(total_step) + '| network_lr:{} '.format(optimizer.param_groups[0]['lr']) + '| Top-1:{} | Top-5:{}'.format(
                top1.avg, top5.avg))
            
        if total_step % 500 == 0 or total_step == args.steps - 1:
            model.eval()
            top1_val = utils_.AvgrageMeter()
            top5_val = utils_.AvgrageMeter()
            with torch.no_grad():
                for step_val, (inputs_val,targets_val) in enumerate(valid_queue):
                    
                    inputs_val = inputs_val.cuda()
                    targets_val = targets_val.cuda()
                    
                    n_val = inputs_val.size(0)
                    logits_val = model(inputs_val)[0]
                    
                    prec_1_val,prec_5_val = utils_.accuracy(logits_val, targets_val, topk=(1, 5))
                    
                    top1_val.update(prec_1_val.data.item(), n_val)
                    top5_val.update(prec_5_val.data.item(), n_val)                
            
            logging.info('step:{} | '.format(total_step) + 'validation | ' + '| Top-1:{} | Top-5:{}'.format(
                top1_val.avg, top5_val.avg))
            model.train()                           
        scheduler.step()   