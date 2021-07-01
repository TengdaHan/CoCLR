import os
import sys
import glob
import msgpack
import lmdb
from io import BytesIO
import torch
from PIL import Image
import pandas as pd 
from tqdm import tqdm 
import random 
import numpy as np 
import math 
import csv
import json

# naming convension:
# {}_2CLIP is for pretraining
# without 2CLIP is for action classification

__all__ = [
    'UCF101LMDB_2CLIP', 'UCF101Flow_LMDB_2CLIP', 'UCF101_2STREAM_LMDB_2CLIP',
    'K400_LMDB_2CLIP', 'K400_Flow_LMDB_2CLIP', 'K400_2STREAM_LMDB_2CLIP',
    'UCF101LMDB', 'UCF101Flow_LMDB',
    'HMDB51LMDB', 'HMDB51Flow_LMDB',
]

# rewrite for yourself:
lmdb_root = '/users/htd/beegfs/DATA/'

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def pil_from_raw_rgb(raw):
    return Image.open(BytesIO(raw)).convert('RGB')

def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content 


class UCF101LMDB_2CLIP(object):
    def __init__(self, root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join(lmdb_root, 'UCF101/ucf101_frame.lmdb'),
                 transform=None, mode='val',
                 num_frames=32, ds=1, which_split=1,
                 window=False, 
                 return_path=False, 
                 return_label=False, 
                 return_source=False):
        self.root = root
        self.db_path = db_path
        self.transform = transform
        self.mode = mode 
        self.num_frames = num_frames
        self.window = window
        self.ds = ds
        self.which_split = which_split
        self.return_label = return_label
        self.return_source = return_source
        self.return_path = return_path

        print('Loading LMDB from %s, split:%d' % (self.db_path, self.which_split))
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.db_length = msgpack.loads(txn.get(b'__len__'))
            self.db_keys = msgpack.loads(txn.get(b'__keys__'))
            self.db_order = msgpack.loads(txn.get(b'__order__'))
        
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]: classes = [i.split(',')[-1].strip() for i in classes]
        print('Frame Dataset from "%s" has #class %d' %(root, len(classes)))

        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}
        
        split_mode = mode
        if mode == 'val': split_mode = 'test'
        video_info = pd.read_csv(os.path.join(root, '%s_split%02d.csv' % (split_mode, which_split)), header=None)
        video_info[2] = video_info[0].str.split('/').str.get(-3)
        video_info[3] = video_info[2]+'/'+video_info[0].str.split('/').str.get(-2)
        assert len(pd.unique(video_info[2])) == self.num_class

        # load video source to id dictionary, 
        # only useful to handle sibling videos in UCF101 pre-training
        if self.return_source:
            self.video_source = read_json(os.path.join(root, 'video_source.json'))
        
        self.get_video_id = dict(zip([i.decode() for i in self.db_order], 
                                     ['%09d'%i for i in range(len(self.db_order))]))

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info), disable=True):
            vpath, vlen, _, _ = row
            if vlen-self.num_frames//2*self.ds-1 <= 0: # allow max padding = half video
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        self.video_subset = self.video_info

    def frame_sampler(self, total):
        if (self.mode == 'test') or self.window: # half overlap - 1
            if total-self.num_frames*self.ds <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames)*self.ds, 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten(0)
        else: # train or val
            if total-self.num_frames*self.ds <= 0: # pad left
                sequence = np.arange(self.num_frames)*self.ds + np.random.choice(range(self.ds),1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start
        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]
        env = self.env
        with env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(self.get_video_id[vname].encode('ascii')))
        
        frame_index = self.double_sampler(vlen)
        seq = [pil_from_raw_rgb(raw[i]) for i in frame_index]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_source:
                source_id = self.video_source[vname.split('/')[-1][0:-4]]
                return seq, source_id, vid
            elif self.return_path:
                return seq, (vid, vpath)
            else:
                return seq, vid
        return seq

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class UCF101LMDB(UCF101LMDB_2CLIP):
    def __init__(self, **kwargs):
        super(UCF101LMDB, self).__init__(**kwargs)
        
    def __getitem__(self, index):
        vpath, vlen, vlabel, vname = self.video_subset.iloc[index]
        env = self.env
        with env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(self.get_video_id[vname].encode('ascii')))
        
        frame_index = self.frame_sampler(vlen)
        seq = [pil_from_raw_rgb(raw[i]) for i in frame_index]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_source:
                source_id = self.video_source[vname.split('/')[-1][0:-4]]
                return seq, source_id, vid
            elif self.return_path:
                return seq, (vid, vpath)
            else:
                return seq, vid
        return seq


class HMDB51LMDB(UCF101LMDB):
    def __init__(self, root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join(lmdb_root, 'HMDB51/hmdb51_frame.lmdb'),
                 **kwargs):
        super(HMDB51LMDB, self).__init__(root=root, db_path=db_path, **kwargs)


class UCF101Flow_LMDB_2CLIP(object):
    def __init__(self, root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join(lmdb_root, 'UCF101/ucf101_tvl1_frame.lmdb'),
                 transform=None, mode='val',
                 num_frames=32, ds=1, which_split=1,
                 return_label=False,
                 return_path=False,
                 return_source=False):
        self.root = root
        self.db_path = db_path
        self.transform = transform
        self.mode = mode 
        self.num_frames = num_frames
        self.ds = ds
        self.which_split = which_split
        self.return_label = return_label
        self.return_source = return_source
        self.return_path = return_path

        print('Loading LMDB from %s, split:%d' % (self.db_path, self.which_split))
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.db_length = msgpack.loads(txn.get(b'__len__'))
            self.db_keys = msgpack.loads(txn.get(b'__keys__'))
            self.db_order = msgpack.loads(txn.get(b'__order__'))
            self.vlen_list = msgpack.loads(txn.get(b'__vlen__'))
        
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]: classes = [i.split(',')[-1].strip() for i in classes]
        print('Frame Dataset from "%s" has #class %d' %(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}
        
        split_mode = mode
        if mode == 'val': split_mode = 'test'
        video_info = pd.read_csv(os.path.join(root, '%s_split%02d.csv' % (split_mode, which_split)), header=None)
        video_info[2] = video_info[0].str.split('/').str.get(-3)
        video_info[3] = video_info[2]+'/'+video_info[0].str.split('/').str.get(-2)
        assert len(pd.unique(video_info[2])) == self.num_class

        # load video source to id dictionary
        if self.return_source:
            self.video_source = read_json(os.path.join(root, 'video_source.json'))

        # check vlen because flow may have different num_frames as rgb.
        vname_list = [i.decode() for i in self.db_order]
        vlen_list_ordered = sorted(list(zip([i.decode() for i in self.db_keys], self.vlen_list)), key=lambda x: x[0])
        vlen_list_ordered = [i[-1] for i in vlen_list_ordered]
        video_info = video_info.merge(pd.DataFrame(zip(vname_list, vlen_list_ordered),columns=[3,4]), left_on=3, right_on=3).dropna()

        self.get_video_id = dict(zip([i.decode() for i in self.db_order], 
                                     ['%09d'%i for i in range(len(self.db_order))]))

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info), disable=True):
            vpath, _, _, _, vlen = row
            if vlen-self.num_frames//2*self.ds-1 <= 0: # allow max padding = half video
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        self.video_subset = self.video_info

    def frame_sampler(self, total):
        if self.mode == 'test': # half overlap - 1
            if total-self.num_frames*self.ds <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames)*self.ds, 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten(0)
        else: # train or val
            if total-self.num_frames*self.ds <= 0: # pad left
                sequence = np.arange(self.num_frames)*self.ds + np.random.choice(range(self.ds),1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start
        return seq_idx


    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, _, vlabel, vname, vlen = self.video_subset.iloc[index]
        env = self.env
        with env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(self.get_video_id[vname].encode('ascii')))
        
        frame_index = self.double_sampler(vlen)
        seq = [pil_from_raw_rgb(raw[i]) for i in frame_index]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_source:
                source_id = self.video_source[vname.split('/')[-1][0:-4]]
                return seq, source_id, vid
            elif self.return_path:
                return seq, (vid, vpath)
            else:
                return seq, vid
        return seq

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class UCF101Flow_LMDB(UCF101Flow_LMDB_2CLIP):
    def __init__(self, **kwargs):
        super(UCF101Flow_LMDB, self).__init__(**kwargs)
        
    def __getitem__(self, index):
        vpath, _, vlabel, vname, vlen = self.video_subset.iloc[index]
        env = self.env
        with env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(self.get_video_id[vname].encode('ascii')))
        
        frame_index = self.frame_sampler(vlen)
        seq = [pil_from_raw_rgb(raw[i]) for i in frame_index]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_source:
                source_id = self.video_source[vname.split('/')[-1][0:-4]]
                return seq, source_id, vid
            elif self.return_path:
                return seq, (vid, vpath)
            else:
                return seq, vid
        return seq


class HMDB51Flow_LMDB(UCF101Flow_LMDB):
    def __init__(self, root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join(lmdb_root, 'HMDB51/hmdb51_tvl1_frame.lmdb'),
                 **kwargs):
        super(HMDB51Flow_LMDB, self).__init__(root=root, db_path=db_path, **kwargs)


class UCF101_2STREAM_LMDB_2CLIP(object):
    def __init__(self, root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path_flow=os.path.join(lmdb_root, 'UCF101/ucf101_tvl1_frame.lmdb'),
                 db_path_rgb=os.path.join(lmdb_root, 'UCF101/ucf101_frame.lmdb'),
                 transform=None, mode='val',
                 num_frames=32, ds=1, which_split=1,
                 return_label=False, 
                 return_path=False,
                 return_source=False):
        self.root = root
        self.db_path_flow = db_path_flow
        self.db_path_rgb = db_path_rgb
        self.transform = transform
        self.mode = mode 
        self.num_frames = num_frames
        self.ds = ds
        self.which_split = which_split
        self.return_label = return_label
        self.return_path = return_path
        self.return_source = return_source

        print('Loading flow LMDB from %s, split:%d' % (self.db_path_flow, self.which_split))
        self.env_flow = lmdb.open(self.db_path_flow, subdir=os.path.isdir(self.db_path_flow),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env_flow.begin(write=False) as txn:
            self.db_length_flow = msgpack.loads(txn.get(b'__len__'))
            self.db_keys_flow = msgpack.loads(txn.get(b'__keys__'))
            self.db_order_flow = msgpack.loads(txn.get(b'__order__'))
            self.vlen_list_flow = msgpack.loads(txn.get(b'__vlen__'))

        print('Loading rgb LMDB from %s, split:%d' % (self.db_path_rgb, self.which_split))
        self.env_rgb = lmdb.open(self.db_path_rgb, subdir=os.path.isdir(self.db_path_rgb),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env_rgb.begin(write=False) as txn:
            self.db_length_rgb = msgpack.loads(txn.get(b'__len__'))
            self.db_keys_rgb = msgpack.loads(txn.get(b'__keys__'))
            self.db_order_rgb = msgpack.loads(txn.get(b'__order__'))
        
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]:
            classes = [i.split(',')[-1].strip() for i in classes]
        print('Two-Stream Dataset from "%s" has #class %d' %(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}
        
        split_mode = mode
        if mode == 'val': split_mode = 'test'
        video_info = pd.read_csv(os.path.join(root, '%s_split%02d.csv' % (split_mode, which_split)), header=None)
        video_info[2] = video_info[0].str.split('/').str.get(-3)
        video_info[3] = video_info[2]+'/'+video_info[0].str.split('/').str.get(-2)
        assert len(pd.unique(video_info[2])) == self.num_class

        # load video source to id dictionary
        self.video_source = read_json(os.path.join(root, 'video_source.json'))

        # check vlen
        vname_list_rgb = [i.decode() for i in self.db_order_rgb]
        vname_list_flow = [i.decode() for i in self.db_order_flow]
        vlen_list_ordered = sorted(list(zip([i.decode() for i in self.db_keys_flow], self.vlen_list_flow)), key=lambda x: x[0])
        vlen_list_ordered = [i[-1] for i in vlen_list_ordered]

        vlen_df_flow = pd.DataFrame(zip(vname_list_flow, vlen_list_ordered), columns=[3,4])
        vlen_df_flow = vlen_df_flow[vlen_df_flow[3].isin(vname_list_rgb)]

        video_info = video_info.merge(vlen_df_flow, left_on=3, right_on=3).dropna()
        video_info[4] = video_info[[1,4]].min(axis=1)

        self.get_video_id_flow = dict(zip([i.decode() for i in self.db_order_flow], 
                                     ['%09d'%i for i in range(len(self.db_order_flow))]))
        self.get_video_id_rgb = dict(zip([i.decode() for i in self.db_order_rgb], 
                                     ['%09d'%i for i in range(len(self.db_order_rgb))]))

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info), disable=True):
            vpath, _, _, _, vlen = row
            if vlen-self.num_frames//2*self.ds-1 <= 0: # allow max padding = half video
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        self.video_subset = self.video_info


    def frame_sampler(self, total):
        if self.mode == 'test': # half overlap - 1
            if total-self.num_frames*self.ds <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames)*self.ds, 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten(0)
        else: # train or val
            if total-self.num_frames*self.ds <= 0: # pad left
                sequence = np.arange(self.num_frames)*self.ds + np.random.choice(range(self.ds),1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start

        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, _, vlabel, vname, vlen = self.video_subset.iloc[index]
        env_rgb = self.env_rgb
        with env_rgb.begin(write=False) as txn:
            raw_rgb = msgpack.loads(txn.get(self.get_video_id_rgb[vname].encode('ascii')))
        env_flow = self.env_flow
        with env_flow.begin(write=False) as txn:
            raw_flow = msgpack.loads(txn.get(self.get_video_id_flow[vname].encode('ascii')))
        
        frame_index = self.double_sampler(vlen)
        seq_rgb = [pil_from_raw_rgb(raw_rgb[i]) for i in frame_index]
        seq_flow = [pil_from_raw_rgb(raw_flow[i]) for i in frame_index]

        assert self.transform is not None 
        seq = self.transform(seq_rgb[0:self.num_frames] + seq_flow[0:self.num_frames] \
                           + seq_rgb[self.num_frames::] + seq_flow[self.num_frames::])
        
        seq1 = seq[0:self.num_frames*2] # rgb, flow
        seq2 = seq[self.num_frames*2::] # rgb, flow
        seq1 = torch.stack(seq1, 1)
        seq2 = torch.stack(seq2, 1)

        if self.return_source:
            source_id = self.video_source[vname.split('/')[-1][0:-4]]
            if self.return_label:
                vid = self.encode_action(vlabel)
                return (seq1, seq2), source_id, vid
            else:
                return (seq1, seq2), source_id

        return (seq1, seq2)

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class KineticsLMDB_2CLIP(object):
    def __init__(self, root, db_path, filename,
                 transform=None, mode='val',
                 num_frames=32, ds=1, window=False, 
                 return_label=False,
                 return_path=False,
                 is_flow=False):
        split_mode = mode
        if mode == 'test': split_mode = 'val'
        self.root = root
        self.db_path = os.path.join(db_path, '%s_%s.lmdb' % (filename, split_mode))
        self.transform = transform
        self.mode = mode 
        self.num_frames = num_frames
        self.window = window
        self.ds = ds
        self.return_label = return_label
        self.return_path = return_path
        self.is_flow = is_flow

        print('Loading LMDB from %s' % self.db_path)
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.db_length = msgpack.loads(txn.get(b'__len__'))
            self.db_keys = msgpack.loads(txn.get(b'__keys__'))
            self.db_order = msgpack.loads(txn.get(b'__order__'))
            if self.is_flow: self.vlen_list = msgpack.loads(txn.get(b'__vlen__'))
        
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]:
            classes = [i.split(',')[-1].strip() for i in classes]
        print('%s Dataset from "%s" has #class %d' %(filename, root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}
        
        video_info = pd.read_csv(os.path.join(root, '%s_split.csv' % split_mode), header=None)
        video_info[2] = video_info[0].str.split('/').str.get(-2)
        video_info[3] = video_info[2]+'/'+video_info[0].str.split('/').str.get(-1)
        video_info = video_info[video_info[2].isin(classes)]

        # load video source to id dictionary
        self.video_source = read_json(os.path.join(root, 'video_source.json'))

        if self.is_flow:
            # check vlen for flow dataset
            vname_list = [i.decode() for i in self.db_order]
            vlen_list_ordered = sorted(list(zip([i.decode() for i in self.db_keys], self.vlen_list)), key=lambda x: x[0])
            vlen_list_ordered = [i[-1] for i in vlen_list_ordered]
            video_info = video_info.merge(pd.DataFrame(zip(vname_list, vlen_list_ordered),columns=[3,4]), left_on=3, right_on=3).dropna()

        self.get_video_id = dict(zip([i.decode() for i in self.db_order], 
                                     ['%09d'%i for i in range(len(self.db_order))]))

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info), disable=True):
            if self.is_flow:
                vpath, _, _, _, vlen = row
            else:
                vpath, vlen, _, _ = row
            if vlen-self.num_frames*self.ds-1 <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)
        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        self.video_subset = self.video_info


    def frame_sampler(self, total):
        if (self.mode == 'test') or self.window: # half overlap - 1
            if total-self.num_frames*self.ds <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames)*self.ds, 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten(0)
        else: # train or val
            if total-self.num_frames*self.ds <= 0: # pad left
                sequence = np.arange(self.num_frames)*self.ds + np.random.choice(range(self.ds),1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start
        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        if self.is_flow:
            vpath, _, vlabel, vname, vlen = self.video_subset.iloc[index]
        else:
            vpath, vlen, vlabel, vname = self.video_subset.iloc[index]
        
        env = self.env
        with env.begin(write=False) as txn:
            raw = msgpack.loads(txn.get(self.get_video_id[vname].encode('ascii')))

        frame_index = self.double_sampler(vlen)
        seq = [pil_from_raw_rgb(raw[i]) for i in frame_index]

        if self.transform is not None: seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.return_label:
            vid = self.encode_action(vlabel)
            if self.return_path:
                return seq, (vid, vpath)
            else:
                return seq, vid
        return seq

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.idx_to_class[action_code]


class K400_LMDB_2CLIP(KineticsLMDB_2CLIP):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join('/users/htd/beegfs/', 'kinetics400-256/frame_full/lmdb/'), 
                 filename='k400_frame',
                 **kwargs):
        super(K400_LMDB_2CLIP, self).__init__(root=root, db_path=db_path, filename=filename, is_flow=False, **kwargs)


class K400_Flow_LMDB_2CLIP(KineticsLMDB_2CLIP):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path=os.path.join('/users/htd/beegfs/', 'kinetics400-256/tvl1_flow/lmdb/'), 
                 filename='k400_tvl1_frame',
                 **kwargs):
        super(K400_Flow_LMDB_2CLIP, self).__init__(root=root, db_path=db_path, filename=filename, is_flow=True, **kwargs)


class Kinetics_2STREAM_LMDB_2CLIP(object):
    def __init__(self, root, db_path_flow, db_path_rgb, filename_flow, filename_rgb,
                 transform=None, mode='val',
                 num_frames=32, ds=1,
                 return_label=False, 
                 return_path=False,
                 return_source=False):
        split_mode = mode
        if mode == 'test': split_mode = 'val'

        self.root = root
        self.db_path_flow = os.path.join(db_path_flow, '%s_%s.lmdb' % (filename_flow, split_mode))
        self.db_path_rgb = os.path.join(db_path_rgb, '%s_%s.lmdb' % (filename_rgb, split_mode))
        self.transform = transform
        self.mode = mode 
        self.num_frames = num_frames
        self.ds = ds
        self.return_label = return_label
        self.return_path = return_path
        self.return_source = return_source

        print('Loading flow LMDB from %s' % self.db_path_flow)
        self.env_flow = lmdb.open(self.db_path_flow, subdir=os.path.isdir(self.db_path_flow),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env_flow.begin(write=False) as txn:
            self.db_length_flow = msgpack.loads(txn.get(b'__len__'))
            self.db_keys_flow = msgpack.loads(txn.get(b'__keys__'))
            self.db_order_flow = msgpack.loads(txn.get(b'__order__'))
            self.vlen_list_flow = msgpack.loads(txn.get(b'__vlen__'))

        print('Loading rgb LMDB from %s' % self.db_path_rgb)
        self.env_rgb = lmdb.open(self.db_path_rgb, subdir=os.path.isdir(self.db_path_rgb),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env_rgb.begin(write=False) as txn:
            self.db_length_rgb = msgpack.loads(txn.get(b'__len__'))
            self.db_keys_rgb = msgpack.loads(txn.get(b'__keys__'))
            self.db_order_rgb = msgpack.loads(txn.get(b'__order__'))
        
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        if ',' in classes[0]:
            classes = [i.split(',')[-1].strip() for i in classes]
        print('Two-Stream Dataset from "%s" has #class %d' %(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}
        
        video_info = pd.read_csv(os.path.join(root, '%s_split.csv' % split_mode), header=None)
        video_info[2] = video_info[0].str.split('/').str.get(-2)
        video_info[3] = video_info[0]
        video_info = video_info[video_info[2].isin(classes)]

        # load video source to id dictionary
        self.video_source = read_json(os.path.join(root, 'video_source.json'))

        # check vlen
        vname_list_rgb = [i.decode() for i in self.db_order_rgb]
        vname_list_flow = [i.decode() for i in self.db_order_flow]
        vlen_list_ordered = sorted(list(zip([i.decode() for i in self.db_keys_flow], self.vlen_list_flow)), key=lambda x: x[0])
        vlen_list_ordered = [i[-1] for i in vlen_list_ordered]

        vlen_df_flow = pd.DataFrame(zip(vname_list_flow, vlen_list_ordered), columns=[3,4])
        vlen_df_flow = vlen_df_flow[vlen_df_flow[3].isin(vname_list_rgb)]

        if video_info.iloc[0][3].split('/').__len__() != 2: # long path to short path
            video_info[3] = video_info[3].str.split('/').str.slice(-2,None,1).str.join('/')

        video_info = video_info.merge(vlen_df_flow, left_on=3, right_on=3).dropna()
        video_info[4] = video_info[[1,4]].min(axis=1)

        self.get_video_id_flow = dict(zip([i.decode() for i in self.db_order_flow], 
                                     ['%09d'%i for i in range(len(self.db_order_flow))]))
        self.get_video_id_rgb = dict(zip([i.decode() for i in self.db_order_rgb], 
                                     ['%09d'%i for i in range(len(self.db_order_rgb))]))

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info), disable=True):
            vpath, _, _, _, vlen = row
            if vlen-self.num_frames*self.ds-1 <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        self.video_subset = self.video_info


    def frame_sampler(self, total):
        if self.mode == 'test': # half overlap - 1
            if total-self.num_frames*self.ds <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames)*self.ds, 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten(0)
        else: # train or val
            if total-self.num_frames*self.ds <= 0: # pad left
                sequence = np.arange(self.num_frames)*self.ds + np.random.choice(range(self.ds),1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start

        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        vpath, _, vlabel, vname, vlen = self.video_subset.iloc[index]
        env_rgb = self.env_rgb
        with env_rgb.begin(write=False) as txn:
            raw_rgb = msgpack.loads(txn.get(self.get_video_id_rgb[vname].encode('ascii')))
        env_flow = self.env_flow
        with env_flow.begin(write=False) as txn:
            raw_flow = msgpack.loads(txn.get(self.get_video_id_flow[vname].encode('ascii')))
        
        frame_index = self.double_sampler(vlen)
        seq_rgb = [pil_from_raw_rgb(raw_rgb[i]) for i in frame_index]
        seq_flow = [pil_from_raw_rgb(raw_flow[i]) for i in frame_index]

        assert self.transform is not None 
        seq = self.transform(seq_rgb[0:self.num_frames] + seq_flow[0:self.num_frames] \
                           + seq_rgb[self.num_frames::] + seq_flow[self.num_frames::])
        
        seq1 = seq[0:self.num_frames*2] # rgb, flow
        seq2 = seq[self.num_frames*2::] # rgb, flow
        seq1 = torch.stack(seq1, 1)
        seq2 = torch.stack(seq2, 1)

        if self.return_source:
            source_id = self.video_source[vname]
            if self.return_label:
                vid = self.encode_action(vlabel)
                return (seq1, seq2), source_id, vid
            else:
                return (seq1, seq2), source_id

        return (seq1, seq2)

    def __len__(self):
        return len(self.video_subset)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class K400_2STREAM_LMDB_2CLIP(Kinetics_2STREAM_LMDB_2CLIP):
    def __init__(self, root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)), 
                 db_path_flow=os.path.join('/users/htd/beegfs/', 'kinetics400-256/tvl1_flow/lmdb/'),
                 db_path_rgb=os.path.join('/users/htd/beegfs/', 'kinetics400-256/frame_full/lmdb/'),
                 filename_flow='k400_tvl1_frame', 
                 filename_rgb='k400_frame',
                 **kwargs):
        super(K400_2STREAM_LMDB_2CLIP, self).__init__(root=root, db_path_flow=db_path_flow, 
            db_path_rgb=db_path_rgb, filename_flow=filename_flow, filename_rgb=filename_rgb, **kwargs)
