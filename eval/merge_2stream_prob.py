import os 
import sys
sys.path.append('../../')
import argparse 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils import data 
from utils.utils import AverageMeter, save_checkpoint, \
write_log, calc_topk_accuracy, Logger, ProgressMeter
import pickle 
import numpy as np 
from tqdm import tqdm 
import math 
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
import json 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob1', default='', type=str)
    parser.add_argument('--prob2', default='', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--mode', default='c', type=str)

    args = parser.parse_args()
    return args

def main(args):
    if args.mode == 'c':
        print('Merge classification probability')
        merge_prob(args)
    elif args.mode == 's':
        print('Merge similarity')
        merge_sim(args)

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def get_action_idx(dataset):
    # dataset: ucf101 or hmdb51 
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..','process_data', 'data')
    if dataset == 'ucf101':
        path = os.path.join(root, 'ucf101_beegfs', 'ClassInd.txt')
    elif dataset == 'hmdb51':
        path = os.path.join(root, 'hmdb51_beegfs', 'ClassInd.txt')
    elif dataset == 'k400':
        path = os.path.join(root, 'k400_beegfs', 'ClassInd.txt')
    action_list = read_file(path)
    if ',' in action_list[0]:
        action_list = [i.split(',')[-1] for i in action_list]
    return action_list


def merge_prob(args):
    first_meter = AverageMeter('acc@1', ':.4f')
    second_meter = AverageMeter('acc@1', ':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')

    with open(args.prob1, 'r') as fp:
        prob_dict1 = json.load(fp)

    with open(args.prob2, 'r') as fp:
        prob_dict2 = json.load(fp)
        
    action_list = get_action_idx(args.dataset)

    # Hammering vs HammerThrow. Order is different
    # action_list_debug = list(prob_dict1.keys())
    # action_list_debug = [i.split('/')[-3] for i in action_list_debug]
    # action_list_debug = sorted(np.unique(action_list_debug))

    action_to_idx = dict(zip(action_list, range(len(action_list))))

    for k in tqdm(prob_dict1.keys(), total=len(prob_dict1)):
        if isinstance(prob_dict1[k], dict):
            prob1 = np.array(prob_dict1[k]['mean_prob'])
        else:
            prob1 = np.array(prob_dict1[k])

        if isinstance(prob_dict2[k], dict):
            prob2 = np.array(prob_dict2[k]['mean_prob'])
        else:
            prob2 = np.array(prob_dict2[k])
                        
        if args.dataset == 'k400':
            label = action_to_idx[k.split('/')[-2]]
        else:
            label = action_to_idx[k.split('/')[-3]]
        prob = (prob1 + prob2) / 2
        first_meter.update(np.int(np.argmax(prob1, axis=-1) == label))
        second_meter.update(np.int(np.argmax(prob2, axis=-1) == label))
        top1_meter.update(np.int(np.argmax(prob, axis=-1) == label))
    
    print('merged accuracy: %.6f + %.6f => %.6f' % (first_meter.avg, second_meter.avg, top1_meter.avg))
    sys.exit()


def merge_sim(args):
    vname1_train_feat = torch.load(os.path.join(args.prob1, '%s_train_feature.pth.tar' % args.dataset), 
        map_location='cpu')
    vname1_test_feat = torch.load(os.path.join(args.prob1, '%s_test_feature.pth.tar' % args.dataset), 
        map_location='cpu')
    with open(os.path.join(args.prob1, '%s_train_vname.pkl' % args.dataset), 'rb') as fp:
        vname1_train = pickle.load(fp)
    with open(os.path.join(args.prob1, '%s_test_vname.pkl' % args.dataset), 'rb') as fp:
        vname1_test = pickle.load(fp)
    train_label = torch.load(os.path.join(args.prob1, '%s_train_label.pth.tar' % args.dataset),
        map_location='cpu')
    test_label = torch.load(os.path.join(args.prob1, '%s_test_label.pth.tar' % args.dataset),
        map_location='cpu')

    vname1_train = np.squeeze(np.array(vname1_train))
    sort_idx_11 = np.argsort(vname1_train)
    vname1_train = np.array(vname1_train)[sort_idx_11]
    vname1_train_feat = vname1_train_feat[sort_idx_11]
    train_label = train_label[sort_idx_11]

    vname1_test = np.squeeze(np.array(vname1_test))
    sort_idx_12 = np.argsort(vname1_test)
    vname1_test = np.array(vname1_test)[sort_idx_12]
    vname1_test_feat = vname1_test_feat[sort_idx_12]
    test_label = test_label[sort_idx_12]

    vname2_train_feat = torch.load(os.path.join(args.prob2, '%s-f_train_feature.pth.tar' % args.dataset), 
        map_location='cpu')
    vname2_test_feat = torch.load(os.path.join(args.prob2, '%s-f_test_feature.pth.tar' % args.dataset), 
        map_location='cpu')
    with open(os.path.join(args.prob2, '%s-f_train_vname.pkl' % args.dataset), 'rb') as fp:
        vname2_train = pickle.load(fp)
    with open(os.path.join(args.prob2, '%s-f_test_vname.pkl' % args.dataset), 'rb') as fp:
        vname2_test = pickle.load(fp)

    vname2_train = np.squeeze(np.array(vname2_train))
    sort_idx_21 = np.argsort(vname2_train)
    vname2_train = np.array(vname2_train)[sort_idx_21]
    vname2_train_feat = vname2_train_feat[sort_idx_21]

    vname2_test = np.squeeze(np.array(vname2_test))
    sort_idx_22 = np.argsort(vname2_test)
    vname2_test = np.array(vname2_test)[sort_idx_22]
    vname2_test_feat = vname2_test_feat[sort_idx_22]

    if len(vname1_train) < len(vname2_train):
        v2_exist_idx_train = np.isin(vname2_train, vname1_train)
        vname2_train = vname2_train[v2_exist_idx_train]
        vname2_train_feat = vname2_train_feat[v2_exist_idx_train]

        v2_exist_idx_test = np.isin(vname2_test, vname1_test)
        vname2_test = vname2_test[v2_exist_idx_test]
        vname2_test_feat = vname2_test_feat[v2_exist_idx_test]

    if len(vname1_train) > len(vname2_train):
        v1_exist_idx_train = np.isin(vname1_train, vname2_train)
        vname1_train = vname1_train[v1_exist_idx_train]
        vname1_train_feat = vname1_train_feat[v1_exist_idx_train]
        train_label = train_label[v1_exist_idx_train]

        v1_exist_idx_test = np.isin(vname1_test, vname2_test)
        vname1_test = vname1_test[v1_exist_idx_test]
        vname1_test_feat = vname1_test_feat[v1_exist_idx_test]
        test_label = test_label[v1_exist_idx_test]

    assert np.all(vname1_train == vname2_train)
    assert np.all(vname1_test == vname2_test)

    # sim calculation
    # vname1_train_feat = torch.from_numpy(vname1_train_feat)
    # vname2_train_feat = torch.from_numpy(vname2_train_feat)
    # vname1_test_feat = torch.from_numpy(vname1_test_feat)
    # vname2_test_feat = torch.from_numpy(vname2_test_feat)

    vname1_train_feat = preprocess(vname1_train_feat)
    vname2_train_feat = preprocess(vname2_train_feat)
    vname1_test_feat = preprocess(vname1_test_feat)
    vname2_test_feat = preprocess(vname2_test_feat)

    # dot product
    sim1 = vname1_test_feat.matmul(vname1_train_feat.t())
    sim2 = vname2_test_feat.matmul(vname2_train_feat.t())

    sim = sim1 + sim2 

    ks = [1,5,10,20,50]
    NN_acc = []

    for k in ks:
        topkval, topkidx = torch.topk(sim, k, dim=1)
        acc = torch.any(train_label[topkidx] == test_label.unsqueeze(1), dim=1).float().mean().item()
        NN_acc.append(acc)
        print('%dNN acc = %.4f' % (k, acc))

    sys.exit()


def preprocess(tensor):
    center = tensor - tensor.mean(dim=0, keepdim=True)
    return F.normalize(center, p=2, dim=1)


if __name__ == '__main__':
    args = parse_args()
    main(args)
