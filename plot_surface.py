import torch
import copy
import net_plotter
import h5py
import numpy as np
import scheduler
import time
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F
import argparse
import utils_
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def name_surface_file(args, dir_file):
    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    return surf_file + ".h5"

def setup_surface_file(args, surf_file, dir_file):
    try:
        f = h5py.File(surf_file, 'a')
        f['dir_file'] = dir_file
    
        # Create the coordinates(resolutions) at which the function is evaluated
        xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
        f['xcoordinates'] = xcoordinates
    
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
        f.close()
    except:
        pass

    return surf_file

def eval_loss(net, criterion, loader, use_cuda=False):
    correct = 0
    total_loss = 0
    total = 0 # number of samples

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)[0]
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, args):

    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:]

    if loss_key not in f.keys():
        shape = (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        f[loss_key] = losses
        f[acc_key] = accuracies
        
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates)

    print('Computing %d values'% (len(inds)))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    # if args.loss_name == 'mse':
    #     criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net

        net_plotter.set_weights(net, w, d, coord)


        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start
        print(loss_compute_time)

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        print(accuracies)
        f.flush()

    f.close()
    
def plot_2d_contour(surf_file, surf_name='valid_acc', vmin=0.1, vmax=25, vlevel=1.0, show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'valid_err':
        Z = 100 - np.array(f['valid_acc'][:])
    elif surf_name == 'train_err' or surf_name == 'valid_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='viridis', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='viridis', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    # fig = plt.figure()
    # sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
    #                        xticklabels=False, yticklabels=False)
    # sns_plot.invert_yaxis()
    # sns_plot.get_figure().savefig(surf_file + '_' + surf_name + '_2dheat.pdf',
    #                               dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(4.2, 8.0)
    ax.set_xlabel("$l_{1}$")
    ax.set_ylabel("$l_{2}$")
    ax.set_zlabel("Valid error")
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()
    if show: plt.show()

parser = argparse.ArgumentParser('Cifar10')
parser.add_argument('--model_path', type = str, default = 'Eval-Beta_DARTS-20Cells-SVHN-Run-22/best_model.pth')
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--data', type = str, default = '../../data')
parser.add_argument('--dir_file', type = str, default = None)
parser.add_argument('--auto_augment', type = bool, default = False)
parser.add_argument('--cutout', type = bool, default = False)
parser.add_argument('--cutout_length', type = int, default = 0)
parser.add_argument('--xmax', type = float, default = 0.3)
parser.add_argument('--xmin', type = float, default = -0.3)
parser.add_argument('--ymax', type = float, default = 0.3)
parser.add_argument('--ymin', type = float, default = -0.3)
parser.add_argument('--xnum', type = int, default = 10)
parser.add_argument('--ynum', type = int, default = 10)
parser.add_argument('--cuda', type = str, default = True)
parser.add_argument('--gpu', type = int, default = 0)
parser.add_argument('--dataset', type = str, default = 'SVHN', help="[SVHN, CIFAR10]")

args,unparsed = parser.parse_known_args()


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)
    
    model = torch.load(args.model_path, map_location='cpu')
    w = net_plotter.get_weights(model)
    s = copy.deepcopy(model.state_dict())
    
    dir_file = net_plotter.name_direction_file(args)
    net_plotter.setup_direction(args, dir_file, model)
    
    d = net_plotter.load_directions(dir_file)
    
    train_transform, valid_transform = utils_._data_transforms_cifar10(args)
    
    if args.dataset == "CIFAR10":
        train_data = datasets.CIFAR10(root = args.data, train = True, download = True, transform = valid_transform)
        valid_data = datasets.CIFAR10(root = args.data, train = False, download = True, transform = valid_transform)
    else:
        args.data = args.data + '/SVHN'
        train_data = datasets.SVHN(root=args.data,
              transform=train_transform, split="train",
              download=True)
        valid_data = datasets.SVHN(root=args.data,
              transform=valid_transform, split="test", 
              download=True)
        
        train_data.data = train_data.data[0:10000]
        train_data.labels = train_data.labels[0:10000]
    
    # train_queue = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, num_workers=4)
    
    surf_file = name_surface_file(args, dir_file)
    # setup_surface_file(args, surf_file, dir_file)
    
    # crunch(surf_file, model, w, s, d, valid_queue, 'valid_loss', 'valid_acc', args)
    plot_2d_contour(surf_file, surf_name='valid_err', vmin=4.2
                    , vmax=10.0, vlevel=0.2, show=True)
