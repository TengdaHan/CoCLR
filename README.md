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
* [2020.11.17] Upload pretrained weights for UCF101 experiments.
* [2020.10.30] Update "draft" dataloader files, CoCLR code, evaluation code as requested by some researchers. Will check and add detailed instructions later.

### Instruction
Soon.

### Dataset
* TVL1 optical flow for UCF101: [[download]](http://www.robots.ox.ac.uk/~htd/tar/ucf101_flow_lmdb.tar) (tar file, 20.5GB, extract and place them in the same directory)

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
