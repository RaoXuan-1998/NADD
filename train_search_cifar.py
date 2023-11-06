import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import utils_
import torch.utils
from model_search import Architecture, Network_Cifar, Distillartor, Pruner
import model_eval
from pcdarts_model import PC_DARTS
from genotypes import PRIMITIVES
from network_visualization import plot_network
import argparse
import torchvision.datasets as datasets
import logging
import torch.backends.cudnn as cudnn
import time

parser = argparse.ArgumentParser('Cifar10')
parser.add_argument('--save', type = str, default = 'S-CIFAR-01-DA-3op')
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--data', type = str, default = '../../data')
parser.add_argument('--CIFAR100', type = bool, default = False)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--epochs', type = list, default = 120)
parser.add_argument('--computing_node_num', type = int, default = 4)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--grad_clip', type = float, default = 5)
parser.add_argument('--cell_num', type = int, default = 8)
parser.add_argument('--channel', type = int, default = 16)
parser.add_argument('--weight_decay_network', type = float, default = 3e-4)
parser.add_argument('--num_classes', type = int, default = 10)
parser.add_argument('--learning_rate_arch', type = float, default = 0.20)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--learning_rate_network', type = float, default = 0.0005)
parser.add_argument('--learning_rate_network_min',type = float,default = 0.0005)
parser.add_argument('--report_freq', type = float, default = 100)
parser.add_argument('--concat_num', type = int, default = 4)
parser.add_argument('--pin_memory', type = bool, default = True)
parser.add_argument('--num_workers', type = int, default = 0)
parser.add_argument('--affine', type = float, default = False)
parser.add_argument('--auto_augment', type = bool, default = True)
parser.add_argument('--cutout', action='store_false', default = False)
parser.add_argument('--cutout_length', type = int, default = 16)
parser.add_argument('--save_genotype_freq', type = int, default = 1)
parser.add_argument('--option', type = str, default = 'relu')
parser.add_argument('--prune_steps', type = int, default = 1)
parser.add_argument('--baseline', type = float, default = 0.01)
parser.add_argument('--pruner_dict', type = dict, default = {'gamma_0' : 1e-4, 'gamma_max' : 0.10, 'varphi' : 1.01, 'k' : 0.99, 'n_max' : 0.005,
                                                             'tqu_max' : 50, 'tau' : 0.90, 'miu_0' : 1e-1, 'rou' : 1.5, 'xi': 1.0})
parser.add_argument('--warm_up', type = int, default = 0)
parser.add_argument('--distillation', type = bool ,default = True)
parser.add_argument('--knowledge_rate', type = float, default = 1.0)
parser.add_argument('--teacher_channel', type = int, default = 36)
parser.add_argument('--teacher_path', type = str, default = '5F-E-CIFAR-02-A-E35-Autoaugment-CIFAR-R1/epoch_195,model.pth')
parser.add_argument('--teacher_geno_path', type = str, default = 'S-CIFAR-02-Autoaugment/genotypes/epoch_35,geno.npy')
parser.add_argument('--infer_freq', type = int, default = 1)
parser.add_argument('--Flops_min', type = float, default = 250)

args,unparsed = parser.parse_known_args()

if args.CIFAR100 == True:
    args.save = args.save + '-cifar100'
    args.num_classes = 100

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
                teacher_logits, teacher_features = teacher(inputs, obtain_features = True)

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
        
        prec_1,prec_5 = utils_.accuracy(logits,targets,topk=(1, 5))
        
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
        if total_step % (args.prune_steps) == 0:            
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
    
    train_transform, valid_transform = utils_._data_transforms_cifar10(args)
    train_data = datasets.CIFAR10(root = args.data, train = True, download = True, transform = train_transform)
    valid_data = datasets.CIFAR10(root = args.data, train = False, download = True, transform = valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, pin_memory = args.pin_memory,
                                              num_workers = args.num_workers)  
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, pin_memory = args.pin_memory, 
                                              num_workers = args.num_workers)
    
    criterion = nn.CrossEntropyLoss().cuda()   

    network = Network_Cifar(args.channel, args.computing_node_num, args.concat_num, args.cell_num,
                       PRIMITIVES, args.affine, args.baseline, False, args.num_classes, channel_shuffle = False)
        
    student_channel_list = [args.channel*args.concat_num, 2*args.channel*args.concat_num, 4*args.channel*args.concat_num]
    teacher_channel_list = [args.teacher_channel*args.concat_num, 2*args.teacher_channel*args.concat_num, 4*args.teacher_channel*args.concat_num]

    network = network.cuda()
    logging.info("param size = {}MB ".format(utils_.count_parameters_in_MB(network)))
    
    architecture = Architecture(args.cell_num, args.computing_node_num, PRIMITIVES, args.option, args.baseline).cuda()
    
    p_dict = args.pruner_dict
    
    pruner = Pruner(p_dict['gamma_0'], p_dict['gamma_max'], p_dict['varphi'], p_dict['k'], p_dict['n_max'], 
                    p_dict['tqu_max'], p_dict['tau'], p_dict['miu_0'], p_dict['rou'], p_dict['xi'])


    if args.distillation:
        # teacher = PC_DARTS(26, 4, 4, 10, 48, 3, PRIMITIVES, True, True).cuda()

        # teacher_geno = np.load(args.teacher_geno_path, allow_pickle = True).tolist()
        # teacher = model_eval.Network_Cifar(36, True, 10, teacher_geno)
        features_location = [args.cell_num//3 - 1, 2*args.cell_num//3 -1, args.cell_num - 1]
        teacher = torch.load(args.teacher_path, map_location = 'cuda:0')
        x = torch.randn([3,3,32,32]).cuda()
        teacher.eval()
        with torch.no_grad():
            teacher(x, obtain_features = True, features_location = features_location)
        # teacher.load_state_dict(teacher_state)
        # teacher = teacher.cuda()
    
        for para in teacher.parameters():
            para.requires_grad = False           
        
        teacher.eval()

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
    
    scheduler_network = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(args.epochs),
                                                                   eta_min = args.learning_rate_network_min)

    total_step = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time() 

        logging.info('epoch:{} '.format(epoch) + '| network_lr:{} '.format(optimizer.param_groups[0]['lr']))
    
        total_objs, objs, objs_knowledge, entropy, acc, total_step, finish = train(network, architecture, teacher, distillartor, pruner, train_queue, valid_queue, criterion, 
                                                               optimizer, optimizer_arch, epoch, total_step)
        geno = architecture.get_genotypes()
        geno_name = 'epoch_{},geno'.format(epoch)
        
        scheduler_network.step()
        
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)
         
        logging.info('epoch:{} | '.format(epoch) + 'all_loss:{} '.format(total_objs) + 'loss:{} '.format(
            objs) + 'loss_knowledge:{} '.format(objs_knowledge) + 'Top1_acc:{} '.format(acc))
        
        if epoch >= (args.warm_up + 1):
            pruner.plot_fig(args.pictures_path)
        
        if epoch % args.infer_freq == 0 or (epoch + 1) == args.epochs:
            architecture.plot_weights(save_path = args.pictures_path)    
            plot_network(geno, file_name = geno_name, save_path = args.genotypes_path)
            valid_acc,valid_obj = infer(network, architecture, criterion, valid_queue) 
            logging.info('epoch:{} | '.format(epoch) + 'valid_acc:{} '.format(valid_acc))
        
        utils_.save(architecture, os.path.join(args.save, 'architecture.pth'))
        utils_.save(pruner, os.path.join(args.save, 'pruner.pth'))
        utils_.save(network, os.path.join(args.save, 'network.pth'))
        
        geno = np.array(geno)
        np.save(os.path.join(args.genotypes_path, geno_name),geno)
        
        if finish:
            logging.info('finish searching')
            architecture.plot_weights(save_path = args.pictures_path)              
            plot_network(geno, file_name = 'final,geno', save_path = args.genotypes_path)                   
            break
    
if __name__ == '__main__':
    main()   
    


