import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import utils_
import torch.utils
from operations_eval import *
from model_eval import Network_Imagenet
import argparse
import torchvision.datasets as datasets
import logging
import torch.backends.cudnn as cudnn
import time
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('ImageNet')
parser.add_argument('--save', type=str, default='E-Imagenet-10-D-E24-R2')
parser.add_argument('--load_geno_path', type=str, default='S-ImageNet-10-D/pareto_genotypes/epoch_24,pareto_geno.npy')
parser.add_argument('--cells_num', type = int, default = 14)
parser.add_argument('--data',type=str,default='../data',help='location of the data corpus')
parser.add_argument('--batch_size',type=int,default = 512, help='batch_size')
parser.add_argument('--epochs',type=int,default = 250,help='search_epoches')
parser.add_argument('--gpu', type=int, default = 0, help='gpu device id')
parser.add_argument('--seed', type=int, default = 0, help='random seed')
parser.add_argument('--grad_clip', type=float, default = 5, help='gradient clipping')
parser.add_argument('--weight_decay_network', type=float, default = 3e-5, help='weight decay for network parameters')
parser.add_argument('--num_classes',type=int,default = 1000,help='num_classes')
parser.add_argument('--channels',type=int,default = 48,help = 'c_start')
parser.add_argument('--learning_rate_network', type = float, default = 0.25, help='init lenarning rate for network parameters')
parser.add_argument('--momentum',type=float,default = 0.9,help='momentum')
parser.add_argument('--learning_rate_network_min', type = float, default = 0., help='min learning rate for network parameters')
parser.add_argument('--report_freq', type=float, default = 50, help='report frequency')
parser.add_argument('--infer_freq',type=int, default = 1,help = 'infer_freq')
parser.add_argument('--concat_num',type=int,default = 4)
parser.add_argument('--save_point',type=int,default = 20)
parser.add_argument('--auxiliary',type=bool,default = True)
parser.add_argument('--auxiliary_rate',type=float, default = 0.4)
parser.add_argument('--drop_prob',type=float,default = 0.0)
parser.add_argument('--resume',type=bool,default = True)
parser.add_argument('--label_smooth', type=float, default = 0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default = 'linear', help='lr scheduler, linear or cosine')
parser.add_argument('--workers', type=int, default = 8)
parser.add_argument('--auxiliary_weight', type=float, default = 0.4)
parser.add_argument('--parella', type = bool, default = False)

args,unparsed = parser.parse_known_args()
utils_.create_exp_dir(args.save)
utils_.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

args.checkpoint_path = os.path.join(args.save,'checkpoint')
os.mkdir(args.checkpoint_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate_network * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate_network * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_queue, model, criterion, optimizer):
    objs = utils_.AvgrageMeter()
    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()
    batch_time = utils_.AvgrageMeter()
    model.train()

    for step, (inputs, target) in enumerate(train_queue):
        target = target.cuda()
        inputs = inputs.cuda()
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux, features = model(inputs)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils_.accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                         step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils_.AvgrageMeter()
    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()
    model.eval()

    for step, (inputs, target) in enumerate(valid_queue):
        inputs = inputs.cuda()
        target = target.cuda()
        with torch.no_grad():
            logits, _ ,features= model(inputs)
            loss = criterion(logits, target)

        prec1, prec5 = utils_.accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg

def main():
    if not torch.cuda.is_available():
        print('no gpu is available')
        sys.exit(1)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    network_genotype = np.load(args.load_geno_path, allow_pickle = True).tolist()
    network = Network_Imagenet(args.channels, args.auxiliary, args.num_classes, network_genotype).cuda()

    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    # num_gpus = torch.cuda.device_count()
    
    
    #if num_gpus > 1:
    #    network = nn.DataParallel(network).cuda()
    
    logging.info("param size = %fMB", utils_.count_parameters_in_MB(network))
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    
    
    optimizer_network = torch.optim.SGD(
        network.parameters(),
        args.learning_rate_network,
        momentum=args.momentum,
        weight_decay=args.weight_decay_network)
    
    data_dir = os.path.join(args.data, 'imagenet')
    traindir = '/home/amax/raoxuan/pycharm/neural_network/data/imagenet/train'
    validdir = '/home/amax/raoxuan/pycharm/neural_network/data/imagenet/val'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_network, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    lr = args.learning_rate_network
    
    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer_network, epoch)
    
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer_network.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
    
        # if num_gpus > 1:
        #     model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # else:
        #     model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    
    
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, network, criterion_smooth, optimizer_network)
        logging.info('Train_acc: %f', train_acc)
    
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, network, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils_.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer_network.state_dict(),
        }, is_best, args.save)


if __name__ == '__main__':
    main()