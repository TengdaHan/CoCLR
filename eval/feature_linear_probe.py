import os 
import sys
sys.path.append('../')
import argparse 
import pickle 
import numpy as np 
from tqdm import tqdm 
import math 
import json 
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils import data 
from utils.utils import AverageMeter, save_checkpoint, \
write_log, calc_topk_accuracy, Logger, ProgressMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--dirname', default='feature', type=str)

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--schedule', default=[60, 80], nargs='*', 
                        type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--normalize', action='store_true', help='normalize feature')
    parser.add_argument('--final_bn', action='store_true', help='final bn')

    args = parser.parse_args()
    return args


class LP(nn.Module):
    '''create linear probe model'''
    def __init__(self, dim, num_class=10, use_bn=False):
        super(LP, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()

        self.fc = nn.Linear(dim, num_class, bias=True)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return self.fc(x)


class D(object):
    '''create feature dataset'''
    def __init__(self, feature, label, vname=None):
        self.feature = feature
        self.label = label
        self.vname = vname  

    def __getitem__(self, index):
        if self.vname is not None:
            return self.feature[index,:], self.label[index], self.vname[index]
        else:
            return self.feature[index,:], self.label[index]

    def __len__(self):
        return self.label.size(0)


def main(args):
    torch.manual_seed(0) 
    device = torch.device('cpu')
    file_prefix = 'test'

    if os.path.exists(os.path.join(os.path.dirname(args.test), args.dirname, '%s_train_feature.pth.tar' % args.dataset)): 
        train_feature = torch.load(
            os.path.join(os.path.dirname(args.test), 
            args.dirname, '%s_train_feature.pth.tar' % args.dataset)).to(device)
        train_label = torch.load(
            os.path.join(os.path.dirname(args.test), 
            args.dirname, '%s_train_label.pth.tar' % args.dataset)).to(device)
        test_feature = torch.load(
            os.path.join(os.path.dirname(args.test), 
            args.dirname, '%s_test_feature.pth.tar' % args.dataset)).to(device)
        test_label = torch.load(
            os.path.join(os.path.dirname(args.test), 
            args.dirname, '%s_test_label.pth.tar' % args.dataset)).to(device)
        with open(os.path.join(os.path.dirname(args.test), args.dirname, '%s_%s_vname.pkl' 
            % (args.dataset, file_prefix)), 'rb') as fp:
            test_vname = pickle.load(fp)
    else:
        print('feature path does not exist')
        sys.exit(0)

    if args.normalize:
        print('Using normalized feature')
        train_feature = F.normalize(train_feature)
        test_feature = F.normalize(test_feature)

    dim = train_feature.size(-1)
    num_class = train_label.max().item() + 1

    if args.final_bn:
        print('Use final BN layer in network')

    model = LP(dim, num_class, args.final_bn)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    train_set = D(train_feature, train_label)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=0, pin_memory=True)
    test_set = D(test_feature, test_label, test_vname)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=0, pin_memory=True)

    best_acc = 0.0
    best_epoch = 0
    best_model = None

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        if epoch % args.eval_freq == 0:
            val_acc = validate(test_loader, model, criterion, epoch, args)
            is_best = val_acc >= best_acc
            if is_best:
                best_acc = val_acc
                best_epoch = epoch
                best_model = model.state_dict()
                print('Best acc: %.4f' % val_acc)

    print('Final best acc: %.4f' % best_acc)
    print('Save probability ... ')
    model.load_state_dict(best_model)
    validate(test_loader, model, criterion, best_epoch, args, save_prob=True)


def train_one_epoch(data_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    progress = ProgressMeter(len(data_loader), 
        [losses, top1_meter], prefix='Epoch:[{}]'.format(epoch))

    model.train()

    for idx, (feature, label) in tqdm(enumerate(data_loader), total=len(data_loader), disable=args.verbose==0):
        B = feature.size(0)
        feature = feature.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        logit = model(feature)
        loss = criterion(logit, label)

        top1, *_ = calc_topk_accuracy(logit, label, (1,))
        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('train: Epoch[{}] loss {:.4f} acc {:.4f}'.format(epoch, losses.avg, top1_meter.avg))


def validate(data_loader, model, criterion, epoch, args, save_prob=False):
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    progress = ProgressMeter(len(data_loader), 
        [losses, top1_meter], prefix='Epoch:[{}]'.format(epoch))
    stats = {}

    model.eval()

    for idx, (feature, label, vname) in tqdm(enumerate(data_loader), total=len(data_loader), disable=args.verbose==0):
        B = feature.size(0)
        feature = feature.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        logit = model(feature)
        loss = criterion(logit, label)
        top1, *_ = calc_topk_accuracy(logit, label, (1,))
        
        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)

        if save_prob:
            prob = F.softmax(logit, dim=-1).squeeze(1)
            if len(vname) == 1: vname = vname[0]
            for i, (v, p) in enumerate(zip(vname, torch.split(prob, 1, dim=0))):
                p_list = p.tolist()
                stats[v] = p_list

    print('eval[{}] loss {:.4f} acc {:.4f}'.format(epoch, losses.avg, top1_meter.avg))
    if save_prob:
        with open(os.path.join(
            os.path.dirname(args.test), args.dirname, 
            '%s-lp-%s-prob.json' % (os.path.basename(args.test), args.dataset)), 'w') as fp:
            json.dump(stats, fp)
        print('prob saved to %s' % os.path.join(
                os.path.dirname(args.test), args.dirname, 
                '%s-lp-%s-prob.json' % (os.path.basename(args.test), args.dataset)))

    return top1_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = parse_args()
    main(args)