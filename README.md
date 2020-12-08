# CoCLR: Self-supervised Co-Training for Video Representation Learning

![arch](asset/teaser.png)

This repository contains the implementation of:

* InfoNCE (MoCo on videos)
* UberNCE (supervised contrastive learning on videos)
* CoCLR

### Link: 

[[Project Page]](http://www.robots.ox.ac.uk/~vgg/research/CoCLR/)
[[PDF]](http://www.robots.ox.ac.uk/~vgg/publications/2020/Han20b/han20b.pdf)
[[Arxiv]](https://arxiv.org/abs/2010.09709)

### News
* [2020.12.08] Update instructions.
* [2020.11.17] Upload pretrained weights for UCF101 experiments.
* [2020.10.30] Update "draft" dataloader files, CoCLR code, evaluation code as requested by some researchers. Will check and add detailed instructions later.

### Pretrain Instruction

* InfoNCE pretrain on UCF101-RGB
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_nce.py --net s3d --model infonce --moco-k 2048 \
--dataset ucf101-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 300 --schedule 250 280 -j 16
```

* InfoNCE pretrain on UCF101-Flow
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_nce.py --net s3d --model infonce --moco-k 2048 \
--dataset ucf101-f-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 300 --schedule 250 280 -j 16
```

* CoCLR pretrain on UCF101 for one cycle
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_coclr.py --net s3d --topk 5 --moco-k 2048 \
--dataset ucf101-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 100 --schedule 80 --name_prefix Cycle1-FlowMining_ -j 8 \
--pretrain {rgb_infoNCE_checkpoint.pth.tar} {flow_infoNCE_checkpoint.pth.tar}
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_coclr.py --net s3d --topk 5 --moco-k 2048 --reverse \
--dataset ucf101-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 100 --schedule 80 --name_prefix Cycle1-RGBMining_ -j 8 \
--pretrain {flow_infoNCE_checkpoint.pth.tar} {rgb_cycle1_checkpoint.pth.tar} 
```

* InfoNCE pretrain on K400-RGB
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 main_infonce.py --net s3d --model infonce --moco-k 16384 \
--dataset k400-2clip --lr 1e-3 --seq_len 32 --ds 1 --batch_size 32 \
--epochs 300 --schedule 250 280 -j 16
```

* InfoNCE pretrain on K400-Flow
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 teco_fb_main.py --net s3d --model infonce --moco-k 16384 \
--dataset k400-f-2clip --lr 1e-3 --seq_len 32 --ds 1 --batch_size 32 \
--epochs 300 --schedule 250 280 -j 16
```

* CoCLR pretrain on K400 for one cycle
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_coclr.py --net s3d --topk 5 --moco-k 16384 \
--dataset k400-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 50 --schedule 40 --name_prefix Cycle1-FlowMining_ -j 8 \
--pretrain {rgb_infoNCE_checkpoint.pth.tar} {flow_infoNCE_checkpoint.pth.tar}
```
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 main_coclr.py --net s3d --topk 5 --moco-k 16384 --reverse \
--dataset k400-2stream-2clip --seq_len 32 --ds 1 --batch_size 32 \
--epochs 50 --schedule 40 --name_prefix Cycle1-RGBMining_ -j 8 \
--pretrain {flow_infoNCE_checkpoint.pth.tar} {rgb_cycle1_checkpoint.pth.tar} 
```

### Dataset
* TVL1 optical flow for UCF101: [[download]](http://www.robots.ox.ac.uk/~htd/tar/ucf101_flow_lmdb.tar) (tar file, 20.5GB, packed with lmdb)

### Result
Finetune entire network for action classification on UCF101:
![arch](asset/coclr-finetune.png)

### Pretrained Weights

Our models:
* UCF101-RGB-CoCLR: [[download]](http://www.robots.ox.ac.uk/~htd/coclr/CoCLR-ucf101-rgb-128-s3d-ep182.tar) [NN@1=51.8 on UCF101-RGB]
* UCF101-Flow-CoCLR: [[download]](http://www.robots.ox.ac.uk/~htd/coclr/CoCLR-ucf101-flow-128-s3d-epoch109.pth.tar) [NN@1=48.4 on UCF101-Flow]

Baseline models:
* UCF101-RGB-InfoNCE: [[download]](http://www.robots.ox.ac.uk/~htd/coclr/InfoNCE-ucf101-rgb-128-s3d-ep399.pth.tar) [NN@1=33.1 on UCF101-RGB]
* UCF101-Flow-InfoNCE: [[download]](http://www.robots.ox.ac.uk/~htd/coclr/InfoNCE-ucf101-f-128-s3d-ep396.pth.tar) [NN@1=45.2 on UCF101-Flow]

Kinetics400-pretrained models comming soon. 
