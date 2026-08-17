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
    "CustomASLLMDB_2CLIP", "CustomASLFlowLMDB_2CLIP", "CustomASL_2STREAM_LMDB_2CLIP"
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

# CustomASL LMDB dataset for ASL dataset, which is not in the standard format of Kinetics or UCF101
class CustomASLLMDB_2CLIP(object):
    """
    Generic RGB LMDB dataset for customASL datasets.

    Expected LMDB metadata:
        __len__   : number of videos
        __keys__  : LMDB keys
        __order__ : entries in "class_name/video_name" format
        __vlen__  : number of frames for each video

    Each video entry must contain a msgpack-encoded list of JPEG byte arrays.
    """

    def __init__(
        self,
        db_path,
        transform=None,
        mode="train",
        num_frames=32,
        ds=1,
        window=False,
        return_label=False,
        return_path=False,
        minimum_video_ratio=0.5,
    ):
        self.db_path = os.path.abspath(db_path)
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.window = window
        self.return_label = return_label
        self.return_path = return_path
        self.minimum_video_ratio = minimum_video_ratio

        if mode not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported mode: {mode}. "
                "Expected train, val, or test."
            )

        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")

        if ds <= 0:
            raise ValueError("ds must be positive.")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"LMDB does not exist: {self.db_path}"
            )

        print(f"Loading CustomASL LMDB from: {self.db_path}")

        # Read only metadata here, then close the environment.
        # The environment is reopened lazily in each worker process.
        env = self._open_environment()

        try:
            with env.begin(write=False) as txn:
                self.db_length = self._load_required_metadata(
                    txn,
                    b"__len__",
                )
                self.db_keys = self._load_required_metadata(
                    txn,
                    b"__keys__",
                )
                self.db_order = self._load_required_metadata(
                    txn,
                    b"__order__",
                )
                self.vlen_list = self._load_required_metadata(
                    txn,
                    b"__vlen__",
                )
        finally:
            env.close()

        self.db_keys = [
            self._to_bytes(key)
            for key in self.db_keys
        ]

        self.db_order = [
            self._to_string(item)
            for item in self.db_order
        ]

        self.vlen_list = [
            int(video_length)
            for video_length in self.vlen_list
        ]

        self._validate_metadata()

        self.samples = self._build_samples()

        class_names = sorted({
            sample["class_name"]
            for sample in self.samples
        })

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }
        self.idx_to_class = {
            index: class_name
            for class_name, index in self.class_to_idx.items()
        }
        self.num_class = len(class_names)

        for sample in self.samples:
            sample["label"] = self.class_to_idx[
                sample["class_name"]
            ]

        self._env = None

        print(
            f"CustomASL LMDB loaded: "
            f"{len(self.samples)} usable videos, "
            f"{self.num_class} classes."
        )

    def _open_environment(self):
        return lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def _get_environment(self):
        """
        Open LMDB lazily.

        This is safer when PyTorch DataLoader uses multiple worker
        processes because an LMDB environment should not be pickled.
        """
        if self._env is None:
            self._env = self._open_environment()

        return self._env

    @staticmethod
    def _unpack(value):
        """
        Support both old and new msgpack versions.
        """
        if value is None:
            return None

        try:
            return msgpack.loads(value, raw=False)
        except TypeError:
            return msgpack.loads(value)

    def _load_required_metadata(self, transaction, key):
        value = transaction.get(key)

        if value is None:
            raise RuntimeError(
                f"Required LMDB metadata is missing: {key!r}"
            )

        return self._unpack(value)

    @staticmethod
    def _to_string(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")

        return str(value)

    @staticmethod
    def _to_bytes(value):
        if isinstance(value, bytes):
            return value

        return str(value).encode("ascii")

    def _validate_metadata(self):
        if int(self.db_length) != len(self.db_keys):
            raise RuntimeError(
                "__len__ does not match the number of __keys__: "
                f"{self.db_length} != {len(self.db_keys)}"
            )

        if len(self.db_order) != len(self.db_keys):
            raise RuntimeError(
                "__order__ and __keys__ have different lengths: "
                f"{len(self.db_order)} != {len(self.db_keys)}"
            )

        if len(self.vlen_list) != len(self.db_keys):
            raise RuntimeError(
                "__vlen__ and __keys__ have different lengths: "
                f"{len(self.vlen_list)} != {len(self.db_keys)}"
            )

    def _extract_class_name(self, video_name):
        """
        Extract the gloss/action name from:

            gloss_name/video_name
        """
        normalized_name = video_name.replace("\\", "/")
        parts = normalized_name.split("/")

        if len(parts) < 2:
            raise RuntimeError(
                "__order__ entries must use the format "
                f"'class_name/video_name', but received: {video_name}"
            )

        return parts[0]

    def _build_samples(self):
        samples = []
        dropped_count = 0

        minimum_required_frames = max(
            1,
            int(
                self.num_frames
                * self.ds
                * self.minimum_video_ratio
            ),
        )

        for key, video_name, video_length in zip(
            self.db_keys,
            self.db_order,
            self.vlen_list,
        ):
            if video_length < minimum_required_frames:
                dropped_count += 1
                continue

            class_name = self._extract_class_name(video_name)

            samples.append({
                "key": key,
                "video_name": video_name,
                "video_length": video_length,
                "class_name": class_name,
            })

        if not samples:
            raise RuntimeError(
                "No usable videos remain after filtering. "
                f"Required minimum frames: {minimum_required_frames}."
            )

        if dropped_count:
            print(
                f"Dropped {dropped_count} videos shorter than "
                f"{minimum_required_frames} frames."
            )

        return samples

    def frame_sampler(self, total_frames):
        """
        Sample one temporal clip.

        Short videos are left-padded by repeating frame zero,
        following the behavior of the original CoCLR loader.
        """
        clip_span = self.num_frames * self.ds

        if self.mode == "test" or self.window:
            if total_frames <= clip_span:
                sequence = np.arange(self.num_frames) * self.ds
                valid_sequence = sequence[sequence < total_frames]

                indices = np.zeros(
                    self.num_frames,
                    dtype=np.int64,
                )

                if len(valid_sequence) > 0:
                    indices[-len(valid_sequence):] = valid_sequence

                return indices

            available = total_frames - clip_span

            step = max(
                1,
                clip_span // 2 - 1,
            )

            starts = np.arange(
                0,
                available + 1,
                step,
                dtype=np.int64,
            )

            clip_indices = (
                starts[:, None]
                + np.arange(self.num_frames)[None, :] * self.ds
            )

            return clip_indices.reshape(-1)

        if total_frames <= clip_span:
            offset = np.random.randint(0, self.ds)

            sequence = (
                np.arange(self.num_frames) * self.ds
                + offset
            )
            valid_sequence = sequence[
                sequence < total_frames
            ]

            indices = np.zeros(
                self.num_frames,
                dtype=np.int64,
            )

            if len(valid_sequence) > 0:
                indices[-len(valid_sequence):] = valid_sequence

            return indices

        max_start = total_frames - clip_span

        start = np.random.randint(
            0,
            max_start + 1,
        )

        return (
            np.arange(self.num_frames) * self.ds
            + start
        ).astype(np.int64)

    def double_sampler(self, total_frames):
        """
        Sample two clips for CoCLR/InfoNCE pretraining.
        """
        first_clip = self.frame_sampler(total_frames)
        second_clip = self.frame_sampler(total_frames)

        return np.concatenate(
            [first_clip, second_clip]
        )

    def _load_raw_frames(self, key):
        environment = self._get_environment()

        with environment.begin(write=False) as transaction:
            packed_frames = transaction.get(key)

        if packed_frames is None:
            raise KeyError(
                f"Video key was not found in LMDB: {key!r}"
            )

        frames = self._unpack(packed_frames)

        if not isinstance(frames, (list, tuple)):
            raise RuntimeError(
                f"Invalid frame data for LMDB key: {key!r}"
            )

        return frames

    def __getitem__(self, index):
        sample = self.samples[index]

        raw_frames = self._load_raw_frames(
            sample["key"]
        )

        actual_video_length = len(raw_frames)

        if actual_video_length == 0:
            raise RuntimeError(
                f"Video contains no frames: "
                f"{sample['video_name']}"
            )

        # Use the actual value to protect against stale __vlen__ metadata.
        frame_indices = self.double_sampler(
            actual_video_length
        )

        sequence = [
            pil_from_raw_rgb(raw_frames[int(frame_index)])
            for frame_index in frame_indices
        ]

        if self.transform is not None:
            sequence = self.transform(sequence)

        if not isinstance(sequence, torch.Tensor):
            sequence = torch.stack(sequence, dim=1)

        label = sample["label"]

        if self.return_label and self.return_path:
            return sequence, (
                label,
                sample["video_name"],
            )

        if self.return_label:
            return sequence, label

        if self.return_path:
            return sequence, sample["video_name"]

        return sequence

    def __len__(self):
        return len(self.samples)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]

    def __getstate__(self):
        """
        Prevent an open LMDB environment from being pickled.
        """
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CustomASLFlowLMDB_2CLIP(CustomASLLMDB_2CLIP):
    """
    Custom optical-flow LMDB dataset for CoCLR/InfoNCE pretraining.

    Expected LMDB metadata:
        __len__   : number of videos
        __keys__  : LMDB keys
        __order__ : entries in "gloss/video_name" format
        __vlen__  : number of optical-flow frames per video

    Each video entry must contain a msgpack-encoded list of
    three-channel JPEG byte arrays representing optical flow.

    The dataset returns two independently sampled clips:
        [C, 2 * num_frames, H, W]
    """

    def __init__(
        self,
        db_path,
        transform=None,
        mode="train",
        num_frames=32,
        ds=1,
        window=False,
        return_label=False,
        return_path=False,
    ):
        super().__init__(
            db_path=db_path,
            transform=transform,
            mode=mode,
            num_frames=num_frames,
            ds=ds,
            window=window,
            return_label=return_label,
            return_path=return_path,
        )

        print(
            f"Custom ASL flow dataset loaded: "
            f"{len(self)} videos, {self.num_class} classes."
        )



class CustomASLLMDB(CustomASLLMDB_2CLIP):
    """
    Single-clip version for supervised action classification.
    """

    def __getitem__(self, index):
        sample = self.samples[index]

        raw_frames = self._load_raw_frames(
            sample["key"]
        )

        actual_video_length = len(raw_frames)

        if actual_video_length == 0:
            raise RuntimeError(
                f"Video contains no frames: "
                f"{sample['video_name']}"
            )

        frame_indices = self.frame_sampler(
            actual_video_length
        )

        sequence = [
            pil_from_raw_rgb(raw_frames[int(frame_index)])
            for frame_index in frame_indices
        ]

        if self.transform is not None:
            sequence = self.transform(sequence)

        if not isinstance(sequence, torch.Tensor):
            sequence = torch.stack(sequence, dim=1)

        label = sample["label"]

        if self.return_label and self.return_path:
            return sequence, (
                label,
                sample["video_name"],
            )

        if self.return_label:
            return sequence, label

        if self.return_path:
            return sequence, sample["video_name"]

        return sequence


class CustomASL_2STREAM_LMDB_2CLIP(object):
    """
    Two-stream custom ASL dataset for CoCLR training.

    Expected RGB and flow LMDB metadata:
        __len__
        __keys__
        __order__
        __vlen__

    The __order__ entries must use:
        gloss/video_name

    RGB and flow LMDBs must contain matching video names.
    """

    def __init__(
        self,
        db_path_rgb,
        db_path_flow,
        transform=None,
        mode='train',
        num_frames=32,
        ds=1,
        return_label=True,
        return_source=True,
        return_path=False,
    ):
        if mode not in {'train', 'val', 'test'}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        self.db_path_rgb = os.path.abspath(db_path_rgb)
        self.db_path_flow = os.path.abspath(db_path_flow)
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.return_label = return_label
        self.return_source = return_source
        self.return_path = return_path

        if not os.path.exists(self.db_path_rgb):
            raise FileNotFoundError(
                f"RGB LMDB not found: {self.db_path_rgb}"
            )

        if not os.path.exists(self.db_path_flow):
            raise FileNotFoundError(
                f"Flow LMDB not found: {self.db_path_flow}"
            )

        print(f"Loading RGB LMDB: {self.db_path_rgb}")
        self.env_rgb = lmdb.open(
            self.db_path_rgb,
            subdir=os.path.isdir(self.db_path_rgb),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        print(f"Loading flow LMDB: {self.db_path_flow}")
        self.env_flow = lmdb.open(
            self.db_path_flow,
            subdir=os.path.isdir(self.db_path_flow),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env_rgb.begin(write=False) as txn:
            rgb_keys = self._unpack_required(txn, b'__keys__')
            rgb_order = self._unpack_required(txn, b'__order__')
            rgb_vlen = self._unpack_required(txn, b'__vlen__')

        with self.env_flow.begin(write=False) as txn:
            flow_keys = self._unpack_required(txn, b'__keys__')
            flow_order = self._unpack_required(txn, b'__order__')
            flow_vlen = self._unpack_required(txn, b'__vlen__')

        rgb_keys = [self._to_bytes(key) for key in rgb_keys]
        flow_keys = [self._to_bytes(key) for key in flow_keys]

        rgb_order = [self._to_string(name) for name in rgb_order]
        flow_order = [self._to_string(name) for name in flow_order]

        rgb_vlen = [int(length) for length in rgb_vlen]
        flow_vlen = [int(length) for length in flow_vlen]

        rgb_metadata = {
            name: {
                'key': key,
                'length': length,
            }
            for name, key, length in zip(
                rgb_order,
                rgb_keys,
                rgb_vlen,
            )
        }

        flow_metadata = {
            name: {
                'key': key,
                'length': length,
            }
            for name, key, length in zip(
                flow_order,
                flow_keys,
                flow_vlen,
            )
        }

        common_video_names = sorted(
            set(rgb_metadata) & set(flow_metadata)
        )

        if not common_video_names:
            raise RuntimeError(
                "No matching videos were found between RGB and flow LMDBs."
            )

        missing_in_flow = set(rgb_metadata) - set(flow_metadata)
        missing_in_rgb = set(flow_metadata) - set(rgb_metadata)

        if missing_in_flow:
            print(
                f"Warning: {len(missing_in_flow)} RGB videos "
                "do not exist in the flow LMDB."
            )

        if missing_in_rgb:
            print(
                f"Warning: {len(missing_in_rgb)} flow videos "
                "do not exist in the RGB LMDB."
            )

        minimum_frames = max(
            1,
            self.num_frames // 2 * self.ds + 1,
        )

        self.samples = []

        for video_name in common_video_names:
            rgb_length = rgb_metadata[video_name]['length']
            flow_length = flow_metadata[video_name]['length']

            # Use the smaller length so that the same indices are valid
            # for both modalities.
            usable_length = min(rgb_length, flow_length)

            if usable_length < minimum_frames:
                continue

            gloss = self._extract_gloss(video_name)

            self.samples.append({
                'video_name': video_name,
                'gloss': gloss,
                'rgb_key': rgb_metadata[video_name]['key'],
                'flow_key': flow_metadata[video_name]['key'],
                'length': usable_length,
            })

        if not self.samples:
            raise RuntimeError(
                "No usable matching RGB/flow videos remain after filtering."
            )

        class_names = sorted({
            sample['gloss']
            for sample in self.samples
        })

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }

        self.idx_to_class = {
            index: class_name
            for class_name, index in self.class_to_idx.items()
        }

        self.num_class = len(class_names)

        for sample in self.samples:
            sample['label'] = self.class_to_idx[
                sample['gloss']
            ]

        # CoCLR uses source_id to suppress or identify videos from
        # the same original source. For a custom dataset, each video
        # can initially be treated as a unique source.
        self.video_to_source_id = {
            sample['video_name']: index
            for index, sample in enumerate(self.samples)
        }

        print(
            f"Loaded {len(self.samples)} aligned videos "
            f"from {self.num_class} classes."
        )

    @staticmethod
    def _unpack(value):
        if value is None:
            return None

        try:
            return msgpack.loads(value, raw=False)
        except TypeError:
            return msgpack.loads(value)

    def _unpack_required(self, transaction, key):
        value = transaction.get(key)

        if value is None:
            raise RuntimeError(
                f"Missing LMDB metadata: {key!r}"
            )

        return self._unpack(value)

    @staticmethod
    def _to_string(value):
        if isinstance(value, bytes):
            return value.decode('utf-8')

        return str(value)

    @staticmethod
    def _to_bytes(value):
        if isinstance(value, bytes):
            return value

        return str(value).encode('ascii')

    @staticmethod
    def _extract_gloss(video_name):
        normalized = video_name.replace('\\', '/')
        parts = normalized.split('/')

        if len(parts) < 2:
            raise RuntimeError(
                "__order__ entries must use "
                f"'gloss/video_name': {video_name}"
            )

        return parts[0]

    def frame_sampler(self, total):
        clip_span = self.num_frames * self.ds

        if self.mode == 'test':
            if total <= clip_span:
                sequence = np.arange(self.num_frames) * self.ds
                sequence = sequence[sequence < total]

                indices = np.zeros(
                    self.num_frames,
                    dtype=np.int64,
                )
                indices[-len(sequence):] = sequence
                return indices

            available = total - clip_span
            step = max(1, clip_span // 2 - 1)

            starts = np.arange(
                0,
                available + 1,
                step,
                dtype=np.int64,
            )

            indices = (
                starts[:, None]
                + np.arange(self.num_frames)[None, :] * self.ds
            )

            return indices.reshape(-1)

        if total <= clip_span:
            offset = np.random.randint(0, self.ds)

            sequence = (
                np.arange(self.num_frames) * self.ds
                + offset
            )
            sequence = sequence[sequence < total]

            indices = np.zeros(
                self.num_frames,
                dtype=np.int64,
            )
            indices[-len(sequence):] = sequence
            return indices

        max_start = total - clip_span

        start = np.random.randint(
            0,
            max_start + 1,
        )

        return (
            np.arange(self.num_frames) * self.ds
            + start
        ).astype(np.int64)

    def double_sampler(self, total):
        first_clip = self.frame_sampler(total)
        second_clip = self.frame_sampler(total)

        return np.concatenate([
            first_clip,
            second_clip,
        ])

    def _load_video(self, environment, key):
        with environment.begin(write=False) as txn:
            packed = txn.get(key)

        if packed is None:
            raise KeyError(
                f"LMDB key not found: {key!r}"
            )

        frames = self._unpack(packed)

        if not isinstance(frames, (list, tuple)):
            raise RuntimeError(
                f"Invalid video data for key: {key!r}"
            )

        return frames

    def __getitem__(self, index):
        sample = self.samples[index]

        raw_rgb = self._load_video(
            self.env_rgb,
            sample['rgb_key'],
        )

        raw_flow = self._load_video(
            self.env_flow,
            sample['flow_key'],
        )

        usable_length = min(
            len(raw_rgb),
            len(raw_flow),
            sample['length'],
        )

        frame_indices = self.double_sampler(
            usable_length
        )

        rgb_frames = [
            pil_from_raw_rgb(raw_rgb[int(frame_index)])
            for frame_index in frame_indices
        ]

        flow_frames = [
            pil_from_raw_rgb(raw_flow[int(frame_index)])
            for frame_index in frame_indices
        ]

        if self.transform is None:
            raise RuntimeError(
                "A transform is required for two-stream CoCLR training."
            )

        # Required ordering:
        # first RGB clip, first flow clip,
        # second RGB clip, second flow clip.
        transformed = self.transform(
            rgb_frames[:self.num_frames]
            + flow_frames[:self.num_frames]
            + rgb_frames[self.num_frames:]
            + flow_frames[self.num_frames:]
        )

        first_stream_pair = transformed[
            :self.num_frames * 2
        ]
        second_stream_pair = transformed[
            self.num_frames * 2:
        ]

        seq1 = torch.stack(first_stream_pair, dim=1)
        seq2 = torch.stack(second_stream_pair, dim=1)

        output = (seq1, seq2)

        source_id = self.video_to_source_id[
            sample['video_name']
        ]
        label = sample['label']

        if self.return_source and self.return_label:
            return output, source_id, label

        if self.return_source:
            return output, source_id

        if self.return_label:
            return output, label

        if self.return_path:
            return output, sample['video_name']

        return output

    def __len__(self):
        return len(self.samples)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]

    def close(self):
        if getattr(self, 'env_rgb', None) is not None:
            self.env_rgb.close()
            self.env_rgb = None

        if getattr(self, 'env_flow', None) is not None:
            self.env_flow.close()
            self.env_flow = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

