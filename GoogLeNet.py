'''
GoogLeNet pytorch implementation
'Going Deeper with Convolutions' https://arxiv.org/abs/1409.4842
'''

import torch.nn as nn
import torch

class Inception(nn.Module):
    
    def __init__(self, input_channels, num_1x1, num_3x3, num_3x3_reduce,
                 num_5x5, num_5x5_reduce, num_pool, ConvBlock = None):
        super(Inception, self).__init__()
        
        if ConvBlock is None:
            ConvBlock = BasicConv2d
        # 1x1 conv layer aka branch 1
        self.branch1 = ConvBlock(input_channels,
                                 num_1x1,
                                 kernel_size = 1,
                                 stride = 1)
        
        # 3x3 conv layer with a 1x1 reduce layer aka branch 2
        self.branch2 = nn.Sequential(
                ConvBlock(input_channels,
                          num_3x3_reduce,
                          kernel_size = 1,
                          stride = 1),
                ConvBlock(num_3x3_reduce,
                          num_3x3,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                )
        
        # 5x5 conv layer with 1x1 reduce layer aka branch 3
        self.branch3 = nn.Sequential(
                ConvBlock(input_channels,
                          num_5x5_reduce,
                          kernel_size = 1,
                          stride = 1),
                ConvBlock(num_5x5_reduce,
                          num_5x5,
                          kernel_size = 5,
                          stride = 1,
                          padding = 2)
                )
                
        # pooling layer branch aka branch4
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size = 3,
                             stride = 1,
                             padding = 1),
                ConvBlock(input_channels,
                          num_pool,
                          kernel_size = 1,
                          stride = 1)
                )
        
    def forward(self, x):
        
        p1 = self.branch1(x)
        p2 = self.branch2(x)
        p3 = self.branch3(x)
        p4 = self.branch4(x)
        return torch.cat([p1, p2, p3, p4], dim = 1)
    
class BasicConv2d(nn.Module):
    
    def __init__(self, input_size, output_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels = input_size,
                              out_channels = output_size,
                              bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(output_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.functional.relu(x, inplace = True)
        
        

class GoogLeNet(nn.Module):
    
    def __init__(self, ConvBlock = None):
        super(GoogLeNet, self).__init__()
        if ConvBlock is None:
            ConvBlock = BasicConv2d
        self.conv1 = ConvBlock(3, 64, kernel_size = 3, stride = 2, padding = 1)
        self.max1 = nn.MaxPool2d(3, stride = 2, ceil_mode = True)
        
        self.a3 = Inception(64, 64, 128, 96, 32, 16, 32)
        self.b3 = Inception(256, 128, 192, 128, 96, 32, 64)
        self.max2 = nn.MaxPool2d(2, stride = 2, ceil_mode = True)
        
        self.a4 = Inception(480, 192, 208, 96, 48, 16, 64)
        self.b4 = Inception(512, 160, 224, 112, 64, 24, 64)
        self.c4 = Inception(512, 128, 256, 128, 64, 24, 64)
        self.d4 = Inception(512, 112, 288, 144, 64, 32, 64)
        self.e4 = Inception(528, 256, 320, 160, 128, 32, 128)
        self.max3 = nn.MaxPool2d(2 , stride = 2, ceil_mode = True)
        
        self.a5 = Inception(832, 256, 320, 160, 128, 32, 128)
        self.b5 = Inception(832, 384, 384, 192, 128, 48, 128)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, 10)
        
    def forward(self, x):
        # N x 3 x 32 x 32
        x = self.conv1(x)
        # N x 64 x 16 x 16
        x = self.max1(x)
        # N x 64 x 8 x 8
        x = self.a3(x)
        # N x 256 x 8 x 8
        x = self.b3(x)
        # N x 480 x 8 x 8
        x = self.max2(x)
        # N x 480 x 4 x 4
        x = self.a4(x)
        # N x 512 x 4 x 4
        x = self.b4(x)
        # N x 512 x 4 x 4
        x = self.c4(x)
        # N x 512 x 4 x 4
        x = self.d4(x)
        # N x 528 x 4 x 4
        x = self.e4(x)
        # N x 832 x 4 x 4
        x = self.max3(x)
        # N x 832 x 2 x 2
        x = self.a5(x)
        # N x 832 x 2 x 2
        x = self.b5(x)
        # N x 1024 x 2 x 2
        x = self.avg(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc(x)
        # N x 10
        
        return x