import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import utils_
import torch.utils
from model_search import Architecture, Network_ImageNet, Distillartor, Pruner
from genotypes import PRIMITIVES
import torchvision.transforms as transforms
from network_visualization import plot_network
import argparse
import torchvision.datasets as datasets
import logging
import torch.backends.cudnn as cudnn
import time
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser('ImageNet')
parser.add_argument('--save', type = str, default = 'S-ImageNet-10-D')
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--data', type = str, default = '../../data')
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--epoches', type = list, default = 30)
parser.add_argument('--computing_node_num', type = int, default = 4)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--grad_clip', type = float, default = 5)
parser.add_argument('--cell_num', type = int, default = 14)
parser.add_argument('--channel', type = int, default = 20)
parser.add_argument('--weight_decay_network', type = float, default = 3e-4)
parser.add_argument('--num_classes', type = int, default = 1000)
parser.add_argument('--learning_rate_arch', type = float, default = 0.20)
parser.add_argument('--learning_rate_network', type = float, default = 3e-4)
parser.add_argument('--learning_rate_network_min',type=float,default = 3e-4)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--auto_augment', type = bool, default = False)
parser.add_argument('--report_freq', type = float, default = 100)
parser.add_argument('--concat_num', type = int, default = 4)
parser.add_argument('--pin_memory', type = bool, default = True)
parser.add_argument('--num_workers', type = int, default = 16)
parser.add_argument('--affine', type = float, default = False)
parser.add_argument('--cutout', action = 'store_false', default = False)
parser.add_argument('--save_genotype_freq', type = int, default = 1)
parser.add_argument('--option', type = str, default = 'relu')
parser.add_argument('--prune_steps', type = int, default = 1)
parser.add_argument('--baseline', type = float, default = 0.01)
parser.add_argument('--pruner_dict', type = dict, default = {'gamma_0' : 1e-6, 'gamma_max' : 0.001, 'varphi' : 1.01, 'k' : 0.99, 'n_max' : 0.0005,
                                                             'tqu_max' : 300, 'tau' : 0.99, 'miu_0' : 1e-1, 'rou' : 1.5, 'xi': 1.0})
parser.add_argument('--warm_up', type = int, default = 5)
parser.add_argument('--distillation', type = bool ,default = True)
parser.add_argument('--ratio', type = list, default = [1.0, 1.0, 1.0])
parser.add_argument('--knowledge_rate', type = float, default = 1.0)

parser.add_argument('--pretrained_model', type = str, default = 'efficientnet-b0')

parser.add_argument('--infer_freq', type = int, default = 4)
parser.add_argument('--Flops_min', type = float, default = 250)

args,unparsed = parser.parse_known_args()

if not args.debug:
    utils_.create_exp_dir(args.save)
    utils_.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    args.pictures_path = os.path.join(args.save, 'weights')
    args.model_path = os.path.join(args.save, 'models')
    args.genotypes_path = os.path.join(args.save, 'genotypes')
    args.pareto_genotypes_path = os.path.join(args.save, 'pareto_genotypes')
    
    os.mkdir(args.pictures_path)
    os.mkdir(args.model_path)
    os.mkdir(args.genotypes_path)
    os.mkdir(args.pareto_genotypes_path)

def train(network, architecture, teacher, distillartor, pruner, train_queue, valid_queue, criterion, 
          optimizer, optimizer_arch, epoch, total_step):

    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()
    total_objs = utils_.AvgrageMeter()
    objs = utils_.AvgrageMeter()
    objs_knowledge = utils_.AvgrageMeter()
    entropy = utils_.AvgrageMeter()     
    network.train()

    for step,(inputs,targets) in enumerate(train_queue):
        
        total_step = total_step + 1
            
        n = inputs.size(0)            
        inputs = inputs.cuda()
        targets = targets.cuda()

        if args.distillation:   
            with torch.no_grad():
                t_features = teacher.extract_endpoints(inputs)
                teacher_features = [t_features['reduction_3'], t_features['reduction_4'], t_features['reduction_6']]

        logits, logits_aux, features = network(inputs, architecture)
        
        if network.auxiliary:
            loss = criterion(logits, targets) + args.auxiliary_rate*criterion(
                logits_aux, targets)
        else:
            loss = criterion(logits, targets)
        
        if args.distillation:
            loss_knowledge = distillartor(features, teacher_features, args.ratio)
        else:
            loss_knowledge = 0.0            
            
        sparsity = architecture.get_sparsity(grad = True)

        if epoch < args.warm_up:
            all_loss = loss + args.knowledge_rate*loss_knowledge
            all_loss.backward()
        
        else:
            all_loss = (loss + args.knowledge_rate*loss_knowledge)*(1 + pruner.gamma*sparsity) + pruner.gamma*pruner.miu*sparsity
            all_loss.backward()
       
        nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(architecture.parameters(), args.grad_clip)
        
        if args.distillation:
            nn.utils.clip_grad_norm_(distillartor.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer_arch.step()
        optimizer.zero_grad()
        optimizer_arch.zero_grad()
        
        prec_1, prec_5 = utils_.accuracy(logits,targets,topk=(1, 5))
        
        total_objs.update(all_loss.data.item(), n)
        objs.update(loss.data.item(), n)
 
        if args.distillation:
            objs_knowledge.update(loss_knowledge.data.item(), n)
        
        entropy.update(sparsity.data.item(), n)
        
        top1.update(prec_1.data.item(), n)
        top5.update(prec_5.data.item(), n)
        
        finish = False
        if epoch >= args.warm_up:            
            if total_step % args.prune_steps == 0:
                pareto, delate = pruner.prune(architecture, network)
                logging.info(delate)
                if total_step % 1000 == 0:
                    geno = architecture.get_genotypes()
                    name = 'step_{},geno'.format(total_step)
                    plot_network(geno, file_name = name, save_path = args.genotypes_path) 
                    geno = np.array(geno)
                    np.save(os.path.join(args.genotypes_path, name), geno)
                if pareto == True:               
                    logging.info('There exist a pareto genotype')
                    pareto_geno = architecture.get_genotypes()  
                    name = 'epoch_{},pareto_geno'.format(epoch)            
                    plot_network(pareto_geno, file_name = name, save_path = args.pareto_genotypes_path)                
                    pareto_geno = np.array(pareto_geno)   
                    np.save(os.path.join(args.pareto_genotypes_path, name), pareto_geno)
                
                if pruner.sparsity_list[-1] < args.Flops_min:
                    finish = True
                else:
                    finish = False                    
        else:
            if total_step % args.prune_steps == 0:
                pruner.update_sparsity(architecture)
        if total_step % args.prune_steps == 0:            
            logging.info('train:%03d | total_loss:%e | loss:%e | loss_knowledge:%e| spasity:%e | top1:%f | top5:%f | total_step:%03d | gamma: %e | miu: %e',
                             step, total_objs.avg, objs.avg, objs_knowledge.avg, entropy.avg, top1.avg, top5.avg, total_step, pruner.gamma, pruner.miu)    
        architecture.append_weights()
        

    return total_objs.avg, objs.avg, objs_knowledge.avg, entropy.avg, top1.avg, total_step, finish

def infer(network, architecture, criterion, valid_queue):
    
    objs = utils_.AvgrageMeter()
    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()
    network.eval()

    with torch.no_grad():     
        for step,(inputs,targets) in enumerate(valid_queue):            
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            logits, logits_aux, features = network(inputs, architecture)
            loss = criterion(logits, targets)
            
            prec1, prec5 = utils_.accuracy(logits, targets, topk=(1, 5))
            n = inputs.size(0)
            
            objs.update(loss.data.item(),n)
            top1.update(prec1.data.item(),n)
            top5.update(prec5.data.item(),n)

    return top1.avg, objs.avg

def main():
    if not torch.cuda.is_available():
        print('no gpu is available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    cudnn.benchmark = True
    cudnn.enabled = True

    data_dir = os.path.join(args.data, 'imagenet')
#    traindir = os.path.join(data_dir, 'train')
 #   validdir = os.path.join(data_dir, 'val')
    traindir = '/home/amax/raoxuan/pycharm/neural_network/data/imagenet/train'
    validdir = '/home/amax/raoxuan/pycharm/neural_network/data/imagenet/val'
    
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4,
                hue = 0.2),
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
        train_data,
        batch_size = args.batch_size,
        pin_memory = args.pin_memory,
        num_workers = args.num_workers,
        shuffle = True)
    
    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size = args.batch_size,
        pin_memory = args.pin_memory,
        num_workers = args.num_workers)
    
    criterion = nn.CrossEntropyLoss().cuda()   

    network = Network_ImageNet(args.channel, args.computing_node_num, args.concat_num, args.cell_num,
                       PRIMITIVES, args.affine, args.baseline, False, args.num_classes).cuda()
        
    student_channel_list = [args.channel*args.concat_num, 2*args.channel*args.concat_num, 4*args.channel*args.concat_num]
    teacher_channel_list = [40, 112, 1280]

    network = network.cuda()
    logging.info("param size = {}MB ".format(utils_.count_parameters_in_MB(network)))
    
    architecture = Architecture(args.cell_num, args.computing_node_num, PRIMITIVES, args.option, args.baseline).cuda()
    
    p_dict = args.pruner_dict
    
    pruner = Pruner(p_dict['gamma_0'], p_dict['gamma_max'], p_dict['varphi'], p_dict['k'], p_dict['n_max'], 
                    p_dict['tqu_max'], p_dict['tau'], p_dict['miu_0'], p_dict['rou'], p_dict['xi'])

    if args.distillation:
        teacher = EfficientNet.from_pretrained(args.pretrained_model).cuda()

        for para in teacher.parameters():
            para.requires_grad = False           
        
        # teacher.train()

        distillartor = Distillartor(student_channel_list, teacher_channel_list).cuda()  
        
        optimizer = torch.optim.Adam([{'params' : network.parameters()}, {'params' : distillartor.parameters()}],
                                     lr = args.learning_rate_network, betas = (0.5,0.999), weight_decay = args.weight_decay_network)
        
    else:
        teacher = None
        distillartor = None        

        optimizer = torch.optim.Adam([{'params' : network.parameters()}],
                                     lr = args.learning_rate_network, betas = (0.5,0.999), weight_decay = args.weight_decay_network)
    
    optimizer_arch = torch.optim.SGD(architecture.parameters(), args.learning_rate_arch, momentum = args.momentum)

    # optimizer_arch = torch.optim.Adam(architecture.parameters(), lr = args.learning_rate_arch, betas=(0.5,0.999))
    scheduler_network = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(args.epoches),
                                                                   eta_min = args.learning_rate_network_min)

    total_step = 0
    for epoch in range(args.epoches):
        epoch_start = time.time() 

        logging.info('epoch:{} '.format(epoch) + '| network_lr:{} '.format(optimizer.param_groups[0]['lr']))
    
        total_objs, objs, objs_knowledge, entropy, acc, total_step, finish = train(network, architecture, teacher, distillartor, pruner, train_queue, valid_queue, criterion, 
                                                               optimizer, optimizer_arch, epoch, total_step)
        
        # geno = architecture.get_genotypes()
        # geno_name = 'epoch_{},geno'.format(epoch)
        
        scheduler_network.step()
        
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)
         
        logging.info('epoch:{} | '.format(epoch) + 'all_loss:{} '.format(total_objs) + 'loss:{} '.format(
            objs) + 'loss_knowledge:{} '.format(objs_knowledge) + 'Top1_acc:{} '.format(acc))
        
        if epoch >= (args.warm_up + 1):
            pruner.plot_fig(args.pictures_path)
        
        if epoch % args.infer_freq == 0 or (epoch + 1) == args.epoches:
            architecture.plot_weights(save_path = args.pictures_path)    
            # plot_network(geno, file_name = geno_name, save_path = args.genotypes_path)
            valid_acc,valid_obj = infer(network, architecture, criterion, valid_queue) 
            logging.info('epoch:{} | '.format(epoch) + 'valid_acc:{} '.format(valid_acc))
        
        utils_.save(architecture, os.path.join(args.save, 'architecture.pth'))
        utils_.save(pruner, os.path.join(args.save, 'pruner.pth'))
        utils_.save(network, os.path.join(args.save, 'network.pth'))
         
        # geno = np.array(geno)
        # np.save(os.path.join(args.genotypes_path, geno_name),geno)
        
        if finish:
            logging.info('finish searching')
            architecture.plot_weights(save_path = args.pictures_path)       
            # plot_network(geno, file_name = 'final,geno', save_path = args.genotypes_path)               
            break
    
if __name__ == '__main__':
    main()   