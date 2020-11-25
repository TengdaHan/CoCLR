import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
from backbone.select_backbone import select_backbone


class LinearClassifier(nn.Module):
    def __init__(self, num_class=101, 
                 network='resnet50', 
                 dropout=0.5, 
                 use_dropout=True, 
                 use_l2_norm=False,
                 use_final_bn=False):
        super(LinearClassifier, self).__init__()
        self.network = network
        self.num_class = num_class
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm
        self.use_final_bn = use_final_bn
        
        message = 'Classifier to %d classes with %s backbone;' % (num_class, network)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_final_bn: message += ' + final BN'
        print(message)

        self.backbone, self.param = select_backbone(network)
        
        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
        
        if use_dropout:
            self.final_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.num_class))
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(self.param['feature_size'], self.num_class))
        self._initialize_weights(self.final_fc)
        
    def forward(self, block):
        (B, C, T, H, W) = block.shape
        feat3d = self.backbone(block)
        feat3d = F.adaptive_avg_pool3d(feat3d, (1,1,1)) # [B,C,1,1,1]
        feat3d = feat3d.view(B, self.param['feature_size']) # [B,C]

        if self.use_l2_norm:
            feat3d = F.normalize(feat3d, p=2, dim=1)
        
        if self.use_final_bn:
            logit = self.final_fc(self.final_bn(feat3d))
        else:
            logit = self.final_fc(feat3d)

        return logit, feat3d

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)

