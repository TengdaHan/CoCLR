# modified from https://raw.githubusercontent.com/qijiezhao/s3d.pytorch/master/S3DG_Pytorch.py 
import torch.nn as nn
import torch

## pytorch default: torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
## tensorflow s3d code: torch.nn.BatchNorm3d(num_features, eps=1e-3, momentum=0.001, affine=True, track_running_stats=True)

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=False)

        # self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else: # int
            t_stride = stride
            
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),
                              stride=(1,stride,stride),padding=(0,padding,padding), bias=False)
        self.conv2 = nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),
                               stride=(t_stride,1,1),padding=(padding,0,0), bias=False)

        # self.bn1=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        # self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.bn1=nn.BatchNorm3d(out_planes)
        self.bn2=nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # init
        self.conv1.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(mean=0, std=0.01) # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class SepInception(nn.Module):
    def __init__(self, in_planes, out_planes, gating=False):
        super(SepInception, self).__init__()

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [num_out_0_0a, 
        num_out_1_0a, num_out_1_0b,
        num_out_2_0a, num_out_2_0b, 
        num_out_3_0b] = out_planes

        self.branch0 = nn.Sequential(
            BasicConv3d(in_planes, num_out_0_0a, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(num_out_1_0a, num_out_1_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(num_out_2_0a, num_out_2_0b, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum([num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = gating 
        if gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):

    def __init__(self, input_channel=3, gating=False, slow=False):
        super(S3D, self).__init__()
        self.gating = gating 
        self.slow = slow 

        if slow:
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=(1,2,2), padding=3)
        else: # normal
            self.Conv_1a = STConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3) 

        self.block1 = nn.Sequential(self.Conv_1a) # (64, 32, 112, 112)
            
        ###################################

        self.MaxPool_2a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Conv_2b = BasicConv3d(64, 64, kernel_size=1, stride=1) 
        self.Conv_2c = STConv3d(64, 192, kernel_size=3, stride=1, padding=1) 

        self.block2 = nn.Sequential(
            self.MaxPool_2a, # (64, 32, 56, 56)
            self.Conv_2b,    # (64, 32, 56, 56)
            self.Conv_2c)    # (192, 32, 56, 56)

        ###################################
        
        self.MaxPool_3a = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) 
        self.Mixed_3b = SepInception(in_planes=192, out_planes=[64, 96, 128, 16, 32, 32], gating=gating)
        self.Mixed_3c = SepInception(in_planes=256, out_planes=[128, 128, 192, 32, 96, 64], gating=gating)

        self.block3 = nn.Sequential(
            self.MaxPool_3a,    # (192, 32, 28, 28)
            self.Mixed_3b,      # (256, 32, 28, 28)
            self.Mixed_3c)      # (480, 32, 28, 28)

        ###################################
        
        self.MaxPool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = SepInception(in_planes=480, out_planes=[192, 96, 208, 16, 48, 64], gating=gating)
        self.Mixed_4c = SepInception(in_planes=512, out_planes=[160, 112, 224, 24, 64, 64], gating=gating)
        self.Mixed_4d = SepInception(in_planes=512, out_planes=[128, 128, 256, 24, 64, 64], gating=gating)
        self.Mixed_4e = SepInception(in_planes=512, out_planes=[112, 144, 288, 32, 64, 64], gating=gating)
        self.Mixed_4f = SepInception(in_planes=528, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,    # (512, 16, 14, 14)
            self.Mixed_4c,    # (512, 16, 14, 14)
            self.Mixed_4d,    # (512, 16, 14, 14)
            self.Mixed_4e,    # (528, 16, 14, 14)
            self.Mixed_4f)    # (832, 16, 14, 14)

        ###################################
        
        self.MaxPool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = SepInception(in_planes=832, out_planes=[256, 160, 320, 32, 128, 128], gating=gating)
        self.Mixed_5c = SepInception(in_planes=832, out_planes=[384, 192, 384, 48, 128, 128], gating=gating)

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,    # (832, 8, 7, 7)
            self.Mixed_5c)    # (1024, 8, 7, 7)

        ###################################

        # self.AvgPool_0a = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        # self.Dropout_0b = nn.Dropout3d(dropout_keep_prob)
        # self.Conv_0c = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)

        # self.classifier = nn.Sequential(
        #     self.AvgPool_0a,
        #     self.Dropout_0b,
        #     self.Conv_0c)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x 

        

if __name__=='__main__':
    model=S3D(num_classes=400)