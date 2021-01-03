import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

__all__ = [
    'ResNet2d3d', 'r2d3d50', 'r3d50'
]

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)

def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out



class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_final_relu=True):
        super(Bottleneck3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3,1,1), padding=(1,0,0), bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_final_relu=True):
        super(Bottleneck2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu: out = self.relu(out)

        return out



class ResNet2d3d(nn.Module):
    def __init__(self, block, layers, input_channel=3):
        super(ResNet2d3d, self).__init__()
        self.inplanes = 64
        bias = False
        self.conv1 = nn.Conv3d(input_channel, 64, kernel_size=(5,7,7), stride=(2, 2, 2), padding=(2, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=(1,2,2))
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=(1,2,2))
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=(1,2,2), is_final=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if isinstance(stride, int):
                if (block == Bottleneck2d) or (block == BasicBlock2d):
                    customized_stride = (1, stride, stride)
                else:
                    customized_stride = stride
            elif isinstance(stride, tuple):
                customized_stride = stride
                stride = stride[-1]
            else:
                raise NotImplementedError

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False), 
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if is_final: # if is final block, no ReLU in the final output
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)

        return F.relu(x)


# used by CVRL(https://arxiv.org/pdf/2008.03800.pdf)
def r2d3d50(**kwargs):
    '''Constructs a ResNet-50 model. '''
    model = ResNet2d3d([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

# full resnet3d
def r3d50(**kwargs):
    '''Constructs a ResNet-50 model. '''
    model = ResNet2d3d([Bottleneck3d, Bottleneck3d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model



if __name__ == '__main__':
    mymodel = r2d3d50()
    mydata = torch.FloatTensor(4, 3, 16, 224, 224)
    nn.init.normal_(mydata)
    import ipdb; ipdb.set_trace()
    mymodel(mydata)
