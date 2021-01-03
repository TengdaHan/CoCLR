import os
import glob
import math
import pickle
import numpy as np
import torch
from torchvision import transforms
from datetime import datetime
from collections import deque


def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename),
                                   'epoch%s.pth.tar' % str(state['epoch']-gap))
    if not keep_all:
        try: os.remove(last_epoch_path)
        except: pass

    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 5:
            try: os.remove(past_best[0])
            except: pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))

def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()

def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean)==len(std)==3
    inv_mean = [-mean[i]/std[i] for i in range(3)]
    inv_std = [1/i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)

def batch_denorm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1):
    shape = [1]*tensor.dim(); shape[channel] = 3
    dtype = tensor.dtype 
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).view(shape)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).view(shape)
    output = tensor.mul(std).add(mean)
    return output 

def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res

def calc_mask_accuracy(output, target_mask, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk,1,True,True)

    zeros = torch.zeros_like(target_mask).long()
    pred_mask = torch.zeros_like(target_mask).long()

    res = []
    for k in range(maxk):
        pred_ = pred[:,k].unsqueeze(1)
        onehot = zeros.scatter(1,pred_,1)
        pred_mask = onehot + pred_mask # accumulate 
        if k+1 in topk:
            res.append(((pred_mask * target_mask).sum(1)>=1).float().mean(0))
    return res 


def neq_load_customized(model, pretrained_dict, verbose=True):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                print(k)
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def strfdelta(tdelta, fmt):
    d = {"d": tdelta.days}
    d["h"], rem = divmod(tdelta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    return fmt.format(**d)

class Logger(object):
    '''write something to txt file'''
    def __init__(self, path):
        self.birth_time = datetime.now()
        filepath = os.path.join(path, self.birth_time.strftime('%Y-%m-%d-%H:%M:%S')+'.log')
        self.filepath = filepath
        with open(filepath, 'a') as f:
            f.write(self.birth_time.strftime('%Y-%m-%d %H:%M:%S')+'\n')

    def log(self, string):
        with open(self.filepath, 'a') as f:
            time_stamp = datetime.now() - self.birth_time
            f.write(strfdelta(time_stamp,"{d}-{h:02d}:{m:02d}:{s:02d}")+'\t'+string+'\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='null', fmt=':.4f'):
        self.name = name 
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        if n == 0: return
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)


    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = self.dict[key]
            avg_val = np.average(val)
            len_val = len(val)
            std_val = np.std(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val, std_val])
            else:
                self.save_dict[key] = [[avg_val, std_val]]

            print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, title, std_val, len_val))

            total.extend(val)

        self.dict = {}
        avg_total = np.average(total)
        len_total = len(total)
        std_total = np.std(total)
        print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, title, std_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f)

    def __len__(self):
        return self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)