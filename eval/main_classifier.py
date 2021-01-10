import os
import sys
sys.path.append('../')
import argparse
import time
import re
import numpy as np
import random 
import pickle 
from tqdm import tqdm 
from PIL import Image
import json 
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F 

from model.classifier import LinearClassifier
from dataset.lmdb_dataset import *
from utils.utils import AverageMeter, save_checkpoint, \
write_log, calc_topk_accuracy, denorm, batch_denorm, Logger, \
ProgressMeter, neq_load_customized
import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='s3d', type=str)
    parser.add_argument('--model', default='lincls', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--which_split', default=1, type=int)
    parser.add_argument('--seq_len', default=32, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=1, type=int, help='number of video blocks')
    parser.add_argument('--num_fc', default=1, type=int, help='number of fc')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size per GPU')
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.9, type=float, help='dropout')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--train_what', default='last', type=str)
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    
    parser.add_argument('--prefix', default='linclr', type=str)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--retrieval', action='store_true', help='path of model to ucf retrieval')

    parser.add_argument('--dirname', default=None, type=str, help='dirname for feature')
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--five_crop', action='store_true')
    parser.add_argument('--ten_crop', action='store_true')
    
    args = parser.parse_args()
    return args


def main(args):
    if args.gpu is None:
        args.gpu = str(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    device = torch.device('cuda')

    best_acc = 0
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    num_gpu = len(str(args.gpu).split(','))
    args.batch_size = num_gpu * args.batch_size
    print('=> Effective BatchSize = %d' % args.batch_size)
    args.img_path, args.model_path, args.exp_path = set_path(args)
    
    ### classifier model ###
    num_class_dict = {'ucf101':   101, 'hmdb51':   51, 'k400': 400,
                      'ucf101-f': 101, 'hmdb51-f': 51, 'k400-f': 400}
    args.num_class = num_class_dict[args.dataset]

    if args.train_what == 'last': # for linear probe
        args.final_bn = True 
        args.final_norm = True 
        args.use_dropout = False
    else: # for training the entire network
        args.final_bn = False 
        args.final_norm = False 
        args.use_dropout = True

    if args.model == 'lincls':
        model = LinearClassifier(
                    network=args.net, 
                    num_class=args.num_class,
                    dropout=args.dropout,
                    use_dropout=args.use_dropout,
                    use_final_bn=args.final_bn,
                    use_l2_norm=args.final_norm)
    else: 
        raise NotImplementedError

    model.to(device)

    ### optimizer ###
    if args.train_what == 'last':
        print('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else: 
                params.append({'params': param})
    
    elif args.train_what == 'ft':
        print('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr/10})
            else:
                params.append({'params': param})
    
    else: # train all
        params = []
        print('=> [optimizer] train all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})

    if args.train_what == 'last':
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    if args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        raise NotImplementedError
    
    model = torch.nn.DataParallel(model)
    model_without_dp = model.module
    
    ce_loss = nn.CrossEntropyLoss()
    args.iteration = 1

    ### test: higher priority ### 
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            if args.retrieval: # if directly test on pretrained network
                new_dict = {}
                for k,v in state_dict.items():
                    k = k.replace('encoder_q.0.', 'backbone.')
                    new_dict[k] = v
                state_dict = new_dict
            
            try: model_without_dp.load_state_dict(state_dict)
            except: neq_load_customized(model_without_dp, state_dict, verbose=True)

        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0
            print("[Warning] if test random init weights, press c to continue")
            import ipdb; ipdb.set_trace()

        args.logger = Logger(path=os.path.dirname(args.test))
        args.logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
        
        transform_test_cuda = transforms.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)])
        
        if args.retrieval:
            test_retrieval(model, ce_loss, transform_test_cuda, device, epoch, args)
        elif args.center_crop or args.five_crop or args.ten_crop:
            transform = get_transform('test', args)
            test_dataset = get_data(transform, 'test', args)
            test_10crop(test_dataset, model, ce_loss, transform_test_cuda, device, epoch, args)
        else:
            raise NotImplementedError
        
        sys.exit(0)

    ### data ###
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, 'train', args), 'train', args)
    transform_val = get_transform('val', args)
    val_loader = get_dataloader(get_data(transform_val, 'val', args), 'val', args)

    transform_train_cuda = transforms.Compose([
                T.RandomHorizontalFlip(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)]) # ImageNet
    transform_val_cuda = transforms.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)]) # ImageNet
    
    print('===================================')

    ### restart training ### 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            try: model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))
    
    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k,v in state_dict.items():
                k = k.replace('encoder_q.0.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict

            try: model_without_dp.load_state_dict(state_dict)
            except: neq_load_customized(model_without_dp, state_dict, verbose=True)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.pretrain))
            raise NotImplementedError
    
    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # plot tools
    writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.val_plotter = TB.PlotterThread(writer_val)
    args.train_plotter = TB.PlotterThread(writer_train)

    args.logger = Logger(path=args.img_path)
    args.logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    
    # main loop 
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train_one_epoch(train_loader, model, ce_loss, optimizer, transform_train_cuda, device, epoch, args)

        if epoch % args.eval_freq == 0:
            _, val_acc = validate(val_loader, model, ce_loss, transform_val_cuda, device, epoch, args)

            # save check_point
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, 1, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=False)
    
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))

    if args.train_what == 'last':
        model.eval() # totally freeze BN in backbone
    else:
        model.train()

    if args.final_bn:
        model.module.final_bn.train()

    end = time.time()
    tic = time.time()

    def tr(x): # transformation on tensor
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim)\
               .transpose(1,2).contiguous()

    for idx, (input_seq, target) in enumerate(data_loader):
        data_time.update(time.time() - end)
        B = input_seq.size(0)
        input_seq = tr(input_seq.to(device, non_blocking=True))
        target = target.to(device, non_blocking=True)
        
        input_seq = input_seq.squeeze(1) # num_seq is always 1, seqeeze it
        logit, _ = model(input_seq)
        loss = criterion(logit, target)
        top1, top5 = calc_topk_accuracy(logit, target, (1,5))
        
        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)

            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5', top5_meter.local_avg, args.iteration)

        args.iteration += 1

    print('Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))

    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)
    
    args.logger.log('train Epoch: [{0}][{1}/{2}]\t'
                    'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), t=time.time()-tic))

    return losses.avg, top1_meter.avg


def validate(data_loader, model, criterion, transforms_cuda, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    model.eval()

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim)\
               .transpose(1,2).contiguous()

    with torch.no_grad():
        end = time.time()
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            B = input_seq.size(0)
            input_seq = tr(input_seq.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)

            input_seq = input_seq.squeeze(1) # num_seq is always 1, seqeeze it
            logit, _ = model(input_seq)
            loss = criterion(logit, target)
            top1, top5 = calc_topk_accuracy(logit, target, (1,5))

            losses.update(loss.item(), B)
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()
            
    print('Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
          .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    args.val_plotter.add_data('global/loss', losses.avg, epoch)
    args.val_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.val_plotter.add_data('global/top5', top5_meter.avg, epoch)

    args.logger.log('val Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f}\t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter))

    return losses.avg, top1_meter.avg


def test_10crop(dataset, model, criterion, transforms_cuda, device, epoch, args):    
    prob_dict = {}
    model.eval()

    # aug_list: 1,2,3,4,5 = topleft, topright, bottomleft, bottomright, center
    # flip_list: 0,1 = raw, flip
    if args.center_crop:
        print('Test using center crop')
        args.logger.log('Test using center_crop\n')
        aug_list = [5]; flip_list = [0]; title = 'center'
    if args.five_crop: 
        print('Test using 5 crop')
        args.logger.log('Test using 5_crop\n')
        aug_list = [5,1,2,3,4]; flip_list = [0]; title = 'five'
    if args.ten_crop:
        print('Test using 10 crop')
        args.logger.log('Test using 10_crop\n')
        aug_list = [5,1,2,3,4]; flip_list = [0,1]; title = 'ten'
    
    def tr(x):
        B = x.size(0); assert B == 1
        num_test_sample = x.size(2)//(args.seq_len*args.num_seq)
        return transforms_cuda(x)\
        .view(3,num_test_sample,args.num_seq,args.seq_len,args.img_dim,args.img_dim).permute(1,2,0,3,4,5)

    with torch.no_grad():
        end = time.time()
        # for loop through 10 types of augmentations, then average the probability
        for flip_idx in flip_list:
            for aug_idx in aug_list:
                print('Aug type: %d; flip: %d' % (aug_idx, flip_idx))
                if flip_idx == 0:
                    transform = transforms.Compose([
                                A.RandomHorizontalFlip(command='left'),
                                A.FiveCrop(size=(224,224), where=aug_idx),
                                A.Scale(size=(args.img_dim,args.img_dim)),
                                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
                                A.ToTensor()])
                else:
                    transform = transforms.Compose([
                                A.RandomHorizontalFlip(command='right'),
                                A.FiveCrop(size=(224,224), where=aug_idx),
                                A.Scale(size=(args.img_dim,args.img_dim)),
                                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
                                A.ToTensor()])

                dataset.transform = transform
                dataset.return_path = True
                dataset.return_label = True
                test_sampler = data.SequentialSampler(dataset)
                data_loader = data.DataLoader(dataset,
                                              batch_size=1,
                                              sampler=test_sampler,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

                for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    input_seq = tr(input_seq.to(device, non_blocking=True))
                    input_seq = input_seq.squeeze(1) # num_seq is always 1, seqeeze it
                    logit, _ = model(input_seq)

                    # average probability along the temporal window
                    prob_mean = F.softmax(logit, dim=-1).mean(0, keepdim=True)

                    target, vname = target
                    vname = vname[0]
                    if vname not in prob_dict.keys():
                        prob_dict[vname] = {'mean_prob':[],}
                    prob_dict[vname]['mean_prob'].append(prob_mean)

                if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
                    print('center-crop result:')
                    acc_1 = summarize_probability(prob_dict, 
                        data_loader.dataset.encode_action, 'center')
                    args.logger.log('center-crop:')
                    args.logger.log('test Epoch: [{0}]\t'
                        'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                        .format(epoch, acc=acc_1))

            if (title == 'ten') and (flip_idx == 0):
                print('five-crop result:')
                acc_5 = summarize_probability(prob_dict, 
                        data_loader.dataset.encode_action, 'five')
                args.logger.log('five-crop:')
                args.logger.log('test Epoch: [{0}]\t'
                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                    .format(epoch, acc=acc_5))

    print('%s-crop result:' % title)
    acc_final = summarize_probability(prob_dict, 
        data_loader.dataset.encode_action, 'ten')
    args.logger.log('%s-crop:' % title)
    args.logger.log('test Epoch: [{0}]\t'
                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                    .format(epoch, acc=acc_final))
    sys.exit(0)


def summarize_probability(prob_dict, action_to_idx, title):
    acc = [AverageMeter(),AverageMeter()]
    stat = {}
    for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        try:
            action_name = vname.split('/')[-3]
        except:
            action_name = vname.split('/')[-2]
        target = action_to_idx(action_name)
        mean_prob = torch.stack(item['mean_prob'], 0).mean(0)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob, torch.LongTensor([target]).cuda(), (1,5))
        stat[vname] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

    print('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
          .format(acc=acc))

    with open(os.path.join(os.path.dirname(args.test), 
        '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def test_retrieval(model, criterion, transforms_cuda, device, epoch, args):
    accuracy = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
    model.eval()
    
    def tr(x):
        B = x.size(0); assert B == 1
        test_sample = x.size(2)//(args.seq_len*args.num_seq)
        return transforms_cuda(x)\
        .view(3,test_sample,args.num_seq,args.seq_len,args.img_dim,args.img_dim).permute(1,2,0,3,4,5)

    with torch.no_grad():
        transform = transforms.Compose([
                    A.CenterCrop(size=(224,224)),
                    A.Scale(size=(args.img_dim,args.img_dim)),
                    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.3, consistent=True),
                    A.ToTensor()])

        if args.dataset == 'ucf101':
            d_class = UCF101LMDB
        elif args.dataset == 'ucf101-f':
            d_class = UCF101Flow_LMDB
        elif args.dataset == 'hmdb51':
            d_class = HMDB51LMDB
        elif args.dataset == 'hmdb51-f':
            d_class = HMDB51Flow_LMDB

        train_dataset = d_class(mode='train', 
                            transform=transform, 
                            num_frames=args.num_seq*args.seq_len,
                            ds=args.ds,
                            which_split=1,
                            return_label=True,
                            return_path=True)
        print('train dataset size: %d' % len(train_dataset))

        test_dataset = d_class(mode='test', 
                            transform=transform, 
                            num_frames=args.num_seq*args.seq_len,
                            ds=args.ds,
                            which_split=1,
                            return_label=True,
                            return_path=True)
        print('test dataset size: %d' % len(test_dataset))

        train_sampler = data.SequentialSampler(train_dataset)
        test_sampler = data.SequentialSampler(test_dataset)

        train_loader = data.DataLoader(train_dataset,
                                      batch_size=1,
                                      sampler=train_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      sampler=test_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True)
        if args.dirname is None:
            dirname = 'feature'
        else:
            dirname = args.dirname

        if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)): 
            test_feature = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset)).to(device)
            test_label = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset)).to(device)
        else:
            try: os.makedirs(os.path.join(os.path.dirname(args.test), dirname))
            except: pass 

            print('Computing test set feature ... ')
            test_feature = None
            test_label = []
            test_vname = []
            sample_id = 0 
            for idx, (input_seq, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                B = 1
                input_seq = input_seq.to(device, non_blocking=True)
                input_seq = tr(input_seq)
                current_target, vname = target
                current_target = current_target.to(device, non_blocking=True)

                test_sample = input_seq.size(0)
                input_seq = input_seq.squeeze(1)
                logit, feature = model(input_seq)
                if test_feature is None:
                    test_feature = torch.zeros(len(test_dataset), feature.size(-1), device=feature.device)

                test_feature[sample_id,:] = feature.mean(0)
                test_label.append(current_target)
                test_vname.append(vname)
                sample_id += 1

            print(test_feature.size())
            # test_feature = torch.stack(test_feature, dim=0)
            test_label = torch.cat(test_label, dim=0)
            torch.save(test_feature, os.path.join(os.path.dirname(args.test), dirname, '%s_test_feature.pth.tar' % args.dataset))
            torch.save(test_label, os.path.join(os.path.dirname(args.test), dirname, '%s_test_label.pth.tar' % args.dataset))
            with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset), 'wb') as fp:
                pickle.dump(test_vname, fp)


        if os.path.exists(os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)): 
            train_feature = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset)).to(device)
            train_label = torch.load(os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset)).to(device)
        else:
            print('Computing train set feature ... ')
            train_feature = None
            train_label = []
            train_vname = []
            sample_id = 0
            for idx, (input_seq, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                B = 1
                input_seq = input_seq.to(device, non_blocking=True)
                input_seq = tr(input_seq)
                current_target, vname = target
                current_target = current_target.to(device, non_blocking=True)

                test_sample = input_seq.size(0)
                input_seq = input_seq.squeeze(1)
                logit, feature = model(input_seq)
                if train_feature is None:
                    train_feature = torch.zeros(len(train_dataset), feature.size(-1), device=feature.device)

                train_feature[sample_id,:] = feature.mean(0)
                # train_feature[sample_id,:] = feature[:,-1,:].mean(0)
                train_label.append(current_target)
                train_vname.append(vname)
                sample_id += 1
            # train_feature = torch.stack(train_feature, dim=0)
            print(train_feature.size())
            train_label = torch.cat(train_label, dim=0)
            torch.save(train_feature, os.path.join(os.path.dirname(args.test), dirname, '%s_train_feature.pth.tar' % args.dataset))
            torch.save(train_label, os.path.join(os.path.dirname(args.test), dirname, '%s_train_label.pth.tar' % args.dataset))
            with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset), 'wb') as fp:
                pickle.dump(train_vname, fp)

        ks = [1,5,10,20,50]
        NN_acc = []

        # centering
        test_feature = test_feature - test_feature.mean(dim=0, keepdim=True)
        train_feature = train_feature - train_feature.mean(dim=0, keepdim=True)

        # normalize
        test_feature = F.normalize(test_feature, p=2, dim=1)
        train_feature = F.normalize(train_feature, p=2, dim=1)

        # dot product
        sim = test_feature.matmul(train_feature.t())

        torch.save(sim, os.path.join(os.path.dirname(args.test), dirname, '%s_sim.pth.tar' % args.dataset))

        for k in ks:
            topkval, topkidx = torch.topk(sim, k, dim=1)
            acc = torch.any(train_label[topkidx] == test_label.unsqueeze(1), dim=1).float().mean().item()
            NN_acc.append(acc)
            print('%dNN acc = %.4f' % (k, acc))

        args.logger.log('NN-Retrieval on %s:' % args.dataset)
        for k,acc in zip(ks, NN_acc):
            args.logger.log('\t%dNN acc = %.4f' % (k, acc))

        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_test_vname.pkl' % args.dataset), 'rb') as fp:
            test_vname = pickle.load(fp)

        with open(os.path.join(os.path.dirname(args.test), dirname, '%s_train_vname.pkl' % args.dataset), 'rb') as fp:
            train_vname = pickle.load(fp)

        sys.exit(0)


def adjust_learning_rate(optimizer, epoch, args):
    '''Decay the learning rate based on schedule'''
    # stepwise lr schedule
    ratio = 0.1 if epoch in args.schedule else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * ratio


def get_transform(mode, args):
    if mode == 'train':
        transform = transforms.Compose([
            A.RandomSizedCrop(size=224, consistent=True, bottom_area=0.2),
            A.Scale(args.img_dim),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True),
            A.ToTensor(),
        ])

    elif mode == 'val' or mode == 'test':
        transform = transforms.Compose([
            A.RandomSizedCrop(size=224, consistent=True, bottom_area=0.2),
            A.Scale(args.img_dim),
            A.ToTensor(),
        ])
    return transform 


def get_data(transform, mode, args):
    print('Loading data for "%s" mode' % mode)
    if args.dataset == 'ucf101':
        dataset = UCF101LMDB(mode=mode, transform=transform, 
            num_frames=args.seq_len*args.num_seq, ds=args.ds, which_split=args.which_split,
            return_label=True)

    elif args.dataset == 'ucf101-f':
        dataset = UCF101Flow_LMDB(mode=mode, transform=transform, 
            num_frames=args.seq_len*args.num_seq, ds=args.ds, which_split=args.which_split,
            return_label=True)

    elif args.dataset == 'hmdb51':
        dataset = HMDB51LMDB(mode=mode, transform=transform, 
            num_frames=args.seq_len*args.num_seq, ds=args.ds, which_split=args.which_split,
            return_label=True)
    elif args.dataset == 'hmdb51-f':
        dataset = HMDB51Flow_LMDB(mode=mode, transform=transform, 
            num_frames=args.seq_len*args.num_seq, ds=args.ds, which_split=args.which_split,
            return_label=True)
    
    else: 
        raise NotImplementedError
    return dataset 


def get_dataloader(dataset, mode, args):
    print("Creating data loaders")
    train_sampler = data.RandomSampler(dataset) 
    val_sampler = None 

    if mode == 'train':
        data_loader = data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    elif mode == 'val':
        data_loader = data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    elif mode == 'test':
        data_loader = data.DataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-eval-{args.prefix}/{args.dataset}-{args.img_dim}_sp{args.which_split}_{args.model}_{args.net}\
{1}_bs{args.batch_size}_lr{args.lr}_dp{args.dropout}_wd{args.wd}_seq{args.num_seq}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{0}'.format(
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    '_SGD' if args.optim=='sgd' else '_Adam', \
                    args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path


if __name__ == '__main__':
    args = parse_args()
    main(args)
