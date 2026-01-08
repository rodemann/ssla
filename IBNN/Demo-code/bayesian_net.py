import argparse
import os
import shutil
import time
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_variational as resnet
import numpy as np
from data.generate_data import *


# For tracking performance
best_prec1 = 0


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Simple_FeedForward(nn.Module):
    def __init__(self, num_inputs = 2 * 10, num_outputs=10, width = 10):
        super(Simple_FeedForward, self).__init__()

        self.inputs = num_inputs
        self.outputs = num_outputs
        self.hidden_units = width


        self.linear1 = LinearReparameterization(
            in_features = self.inputs,
            out_features = self.hidden_units,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,)


        self.linear2 = LinearReparameterization(
            in_features = self.hidden_units,
            out_features = self.hidden_units,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,)

        self.linear3 = LinearReparameterization(
            in_features = self.hidden_units,
            out_features = self.outputs,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,)


        self.apply(_weights_init)


    def forward(self, x):

        kl_sum = 0
        out, kl = self.linear1(x)
        out = F.relu(out)
        kl_sum += kl

        out, kl = self.linear2(out)
        out = F.relu(out)
        kl_sum += kl

        out, kl = self.linear3(out)
        kl_sum += kl

        return out, kl_sum




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def validate(args, val_loader, model, criterion, epoch, tb_writer=None):

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output, kl = model(input_var)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            mse_loss = criterion(output, target_var)
            scaled_kl = kl / args.batch_size
            #ELBO loss
            loss = mse_loss + scaled_kl

            output = output.float()
            loss = loss.float()


            if i % args.print_freq == 0:
                print("At epoch - ", epoch, " for batch - ", i, " loss - ", loss)



    return loss

def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch):


    # switch to train mode
    model.train()

    i = 0
    for i, (input, target) in enumerate(train_loader):


        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
        else:
            target = target.cpu()
            input_var = input.cpu()
            target_var = target

        if args.half:
            input_var = input_var.half()


        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(input_var)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)

        kl = torch.mean(torch.stack(kl_), dim=0)
        mean_sq_loss = criterion(output, target_var)
        scaled_kl = kl / args.batch_size

        #ELBO loss
        loss = mean_sq_loss + scaled_kl


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()


        if i % args.print_freq == 0:
            print("At epoch - ", epoch, " for batch - ", i, " loss - ", loss)

def compute_bayesian_probability(predictions):
    '''
    Input : A tensor with the following shape : ( mc_samples, data index, logits_across class )
    Output : For each data sample, and possible class label, the probability. (data_index, class_label)
    '''


    local_predictions = predictions.cpu().detach().numpy()
    test_set_count = predictions.shape[1]
    mc_samples = predictions.shape[0]
    labels_count = predictions.shape[2]

    output_value_stats = []
    for data_index in range(test_set_count):

        prediction_vector = np.zeros(mc_samples)

        for sample in range(mc_samples):
            prediction_vector[sample] = local_predictions[sample][data_index][0]


        mean_val = np.mean(prediction_vector)
        variance_val = np.var(prediction_vector)
        result_dict = {'mean':mean_val, 'variance':variance_val}
        output_label_stats.append(result_dict)


    return output_label_stats


def main(args):

    global best_prec1

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model = Simple_FeedForward(20, 1, args.width)
    model = model.double()

    # Resnet 20
    # model = ResNet(BasicBlock, [3, 3, 3], class_count)


    if torch.cuda.is_available():
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)
    else:
        model.cpu()
        torch.manual_seed(args.seed)
        # torch.manual_seed_all(args.seed)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    diabetes_train_dataset = diabetes_loader(train = True)
    train_loader = DataLoader(diabetes_train_dataset, batch_size = args.batch_size, shuffle=False, num_workers = 1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss().cpu()

    if args.half:
        model.half()
        criterion.half()



    if args.evaluate:
        diabetes_test_dataset = diabetes_loader(train = False, test = True)
        diabetes_test_loader = DataLoader(diabetes_test_dataset,
        batch_size = args.batch_size, shuffle=False, num_workers = 1)
        validate(diabetes_test_loader, model, criterion)
        return

    if args.mode == 'train':

        # Validation is happening in the training data now
        # val_loader = diabetes_loader(train = True, test = False)

        for epoch in range(args.start_epoch, args.epochs):

            lr = args.lr
            if (epoch >= 80 and epoch < 120):
                lr = 0.1 * args.lr
            elif (epoch >= 120 and epoch < 160):
                lr = 0.01 * args.lr
            elif (epoch >= 160 and epoch < 180):
                lr = 0.001 * args.lr
            elif (epoch >= 180):
                lr = 0.0005 * args.lr

            optimizer = torch.optim.Adam(model.parameters(), lr)

            # train for one epoch
            # print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(args, train_loader, model, criterion, optimizer, epoch)

            # No vaidation happening
            # prec1 = validate(args, val_loader, model, criterion, epoch,)

            # is_best = prec1 > best_prec1
            # best_prec1 = max(prec1, best_prec1)


            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                },
                True,
                filename=os.path.join(
                    args.save_dir,
                    'bayesian_pancreas_model.pth'))

    elif args.mode == 'test':

        filename=os.path.join(args.save_dir, 'bayesian_pancreas_model.pth')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])


        test_dataset = diabetes_loader(train = False, test = True)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers = 1)
        pred_probs_mc = []
        output_list = []
        target_list = []
        model.eval()
        total_error = 0.0

        with torch.no_grad():

            len_testset = 0
            # for data, target in test_loader:
            for i, (data, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                else:
                    data, target = data.cpu(), target.cpu()
                output_mc = []
                for mc_run in range(args.num_monte_carlo):
                    output, _ = model.forward(data)
                    output_mc.append(output.cpu().detach().numpy())

                mean_val = np.mean(output_mc, axis = 0)
                variance_val = np.var(output_mc, axis = 0)
                result_dict = {'mean':mean_val, 'variance':variance_val}

                error = np.absolute(mean_val - target.cpu().detach().numpy())
                len_testset += len(error)

                total_error += np.sum(error)
                output_list.append(result_dict)


        print("Average error - ", total_error / len_testset)
        return output_list


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Imprecise Bayesian Neural Network')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='seed for BNN sample')
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--width',
                        default=10,
                        type=int,
                        metavar='N',
                        help='width of NN layers')
    parser.add_argument('-b',
                        '--batch-size',
                        default=128,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.001,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=5e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 20)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--half',
                        dest='half',
                        action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir',
                        dest='save_dir',
                        help='The directory used to save the trained models',
                        default='./checkpoint/bayesian',
                        type=str)
    parser.add_argument('--save-every',
                        dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int,
                        default=10)
    parser.add_argument('--mode', type=str, required=True, help='train | test | test_ibnn')
    parser.add_argument('--num_monte_carlo',
                        type=int,
                        default=20,
                        metavar='N',
                        help='number of Monte Carlo samples to be drawn during inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=5,
                        metavar='N',
                        help='number of Monte Carlo runs during training')
    parser.add_argument('--tensorboard',
                        type=bool,
                        default=True,
                        metavar='N',
                        help='use tensorboard for logging and visualization of training progress')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./logs/pancreas/bayesian',
                        metavar='N',
                        help='use tensorboard for logging and visualization of training progress')
    parser.add_argument('--datapath',
                        type = str,
                        default = './Datasets/CIFAR/CIFAR/')



    args = parser.parse_args()
    main(args)
