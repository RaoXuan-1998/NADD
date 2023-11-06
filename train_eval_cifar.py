import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import utils_
import torch.utils
from operations_eval import *
from model_eval import Network_Cifar
import argparse
import torchvision.datasets as datasets
import logging
import torch.backends.cudnn as cudnn
import time
from network_visualization import plot_network
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser('Cifar10')
parser.add_argument('--save', type = str, default = 'E-CIFAR-02-A-E43-CIFAR-R1')
parser.add_argument('--load_geno_path', type = str, default = 'S-CIFAR-02-Autoaugment/pareto_genotypes/epoch_43,pareto_geno.npy')
parser.add_argument('--cells_num', type = int, default = 14)
parser.add_argument('--CIFAR100', type = bool, default = False)
parser.add_argument('--data', type = str, default = '../../data')
parser.add_argument('--batch_size', type = int, default = 96)
parser.add_argument('--epoches', type = int, default = 600)
parser.add_argument('--gpu', type = int, default = 1)
parser.add_argument('--seed', type = int, default = 31)
parser.add_argument('--grad_clip', type = float, default = 5)
parser.add_argument('--weight_decay_network', type = float, default=3e-4)
parser.add_argument('--num_classes', type = int, default = 10)
parser.add_argument('--channels', type = int, default = 36)
parser.add_argument('--learning_rate_network', type = float, default = 0.025)
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--learning_rate_network_min', type = float, default = 0.0)
parser.add_argument('--auto_augment', default = False)
parser.add_argument('--cutout', default = True)
parser.add_argument('--cutout_length', type = int, default = 16)
parser.add_argument('--report_freq', type = float, default = 200)
parser.add_argument('--stem_multiplier', type = int, default = 3)
parser.add_argument('--infer_freq', type = int, default = 1, help = 'infer_freq')
parser.add_argument('--load_model_path', type = str, default = 'cdarts_v2\epoch_465,model.pt')
parser.add_argument('--intermediate_nodes_num', type = int, default = 4)
parser.add_argument('--concat_num', type = int, default = 4)
parser.add_argument('--save_point', type = int, default = 20)
parser.add_argument('--auxiliary', type = bool, default = True)
parser.add_argument('--auxiliary_rate', type = float, default = 0.4)
parser.add_argument('--drop_prob', type = float, default = 0.30)
parser.add_argument('--resume', type = bool, default = True)

args,unparsed = parser.parse_known_args()

if args.CIFAR100 == True:
    args.load_path = args.load_path + '-cifar100'
    args.save = args.save + '-cifar100'
    args.num_classes = 100
    args.weight_decay_network = 5e-4
utils_.create_exp_dir(args.save)
utils_.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

args.checkpoint_path = os.path.join(args.save,'checkpoint')
os.mkdir(args.checkpoint_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = log_format, datefmt = '%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

    if args.CIFAR100 == True:
        train_data = datasets.CIFAR100(root = args.data, train = True, download = True, transform = train_transform)
        valid_data = datasets.CIFAR100(root = args.data, train = False, download = True, transform = valid_transform)
    else:
        train_data = datasets.CIFAR10(root = args.data, train = True, download = True, transform = train_transform)
        valid_data = datasets.CIFAR10(root = args.data, train = False, download = True, transform = valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size)

    network_genotype = np.load(args.load_geno_path, allow_pickle=True).tolist()

    def calculate_operator_num(geno):
        total_number = 0
        for cell_order in range(geno['cell_num']):
            for intermediate_node_order in range(geno['computing_node_num']):
                node_name = 'cell_{},node_{}'.format(cell_order, intermediate_node_order + 2)
                node_geno = geno['primitives_finally'][node_name]
                total_number = total_number + len(node_geno)
        darts_num = geno['cell_num']*2*4
        return total_number, darts_num

    operator_num, darts_num = calculate_operator_num(network_genotype)

    drop_prob_max = args.drop_prob + (0.15/112)*(operator_num - darts_num)


    # plot_network(network_genotype)
    

    network = Network_Cifar(args.channels, args.auxiliary, args.num_classes, network_genotype)
    count_flops(network, 32)
    
    network = network.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()

    logging.info("param size = %fMB", utils_.count_parameters_in_MB(network))

    optimizer_network = torch.optim.SGD(
        network.parameters(),
        args.learning_rate_network,
        momentum=args.momentum,
        weight_decay=args.weight_decay_network)

    scheduler_network = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_network,float(args.epoches))

    best_valid_acc = 0.

    for epoch in range(args.epoches):

        logging.info('epoch %d', epoch)
        epoch_start = time.time()
        logging.info('epoch:{} '.format(epoch) + 'network_lr:{} '.format(optimizer_network.param_groups[0]['lr']))

        drop_prob = drop_prob_max*epoch/args.epoches

        checkpoint = {'epoch':epoch,
                      'model_state_dict':network.state_dict(),
                      'optimizer_state_dict':optimizer_network.state_dict(),
                      'scheduler_state_dict':scheduler_network.state_dict()}

        torch.save(checkpoint,os.path.join(args.save,'checkpoint.pth'))

        network.update_drop_prob(drop_prob)

        train_acc, train_obj = train(network, train_queue, criterion, optimizer_network, epoch)

        scheduler_network.step()

        logging.info('train_acc %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logging.info('epoch time: %ds.', epoch_duration)

        valid_acc,valid_obj = infer(valid_queue, network, criterion)

        logging.info('epoch:{}  '.format(epoch) + 'valid_acc:{}'.format(valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            utils_.save(network,os.path.join(args.save, 'best_model.pth'))
        logging.info('best_epoch:{} '.format(best_epoch) + 'best_valid_acc:{}'.format(best_valid_acc))

        if epoch % 5 == 0 or (epoch + 1) == args.epoches:
            utils_.save(network, os.path.join(args.save, 'epoch_{},model.pth'.format(epoch)))

def train(network, train_queue, criterion, optimizer_network, epoch):

    objs = utils_.AvgrageMeter()
    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()

    network.train()

    for step, (inputs, targets) in enumerate(train_queue):

        n = inputs.size(0)
        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking = True)

        logits,logits_aux = network(inputs)
        if args.auxiliary:
            loss = criterion(logits, targets) + args.auxiliary_rate*criterion(logits_aux, targets)
        else:
            loss = criterion(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)

        optimizer_network.step()
        optimizer_network.zero_grad()

        prec_1, prec_5 = utils_.accuracy(logits, targets, topk=(1, 5))

        objs.update(loss.data.item(), n)
        top1.update(prec_1.data.item(), n)
        top5.update(prec_5.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('loss: '+ str(loss.data.item()))
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, network, criterion):

    objs = utils_.AvgrageMeter()
    top1 = utils_.AvgrageMeter()
    top5 = utils_.AvgrageMeter()
    network.eval()

    with torch.no_grad():
        for step,(inputs,targets) in enumerate(valid_queue):
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
            logits, logits_aux = network(inputs)
            loss = criterion(logits, targets)
            prec1, prec5 = utils_.accuracy(logits, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)
    return top1.avg, objs.avg

if __name__ == '__main__':
    main()