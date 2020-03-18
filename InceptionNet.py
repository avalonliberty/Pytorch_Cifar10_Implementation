'''
Implementation of InceptionNet
Rethinking the Inception Architecture for Computer Vision<https://arxiv.org/abs/1512.00567>
'''
import torch.nn as nn
import torch

class Basic_Conv(nn.Module):
    
    def __init__(self, input_channel, output_channel, kernel_size, **kwargs):
        #Type : (int, int, Optional)
        super(Basic_Conv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #Type : (Tensor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
    
class InceptionA(nn.Module):
    
    def __init__(self, input_channel, pooling_channel):
        #Type : (int, int)
        super(InceptionA, self).__init__()
        self.branch1_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        self.branch1_3x3_1 = Basic_Conv(64, 96, kernel_size = 3, padding = 1)
        self.branch1_3x3_2 = Basic_Conv(96, 96, kernel_size = 3, padding = 1)
        
        self.branch2_1x1 = Basic_Conv(input_channel, 48, kernel_size = 1)
        self.branch2_3x3 = Basic_Conv(48, 64, kernel_size = 3, padding = 1)
        
        self.branch3_pool = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.branch3_1x1 = Basic_Conv(input_channel, pooling_channel, kernel_size = 1)
        
        self.branch4_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        
    def forward(self, x):
        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_3x3_1(branch1)
        branch1 = self.branch1_3x3_2(branch1)
        
        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3(branch2)
        
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_1x1(branch3)
        
        branch4 = self.branch4_1x1(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim = 1)
    
class InceptionB(nn.Module):
    
    def __init__(self, input_channel, channel_7x7):
        #Type : (int, int)
        super(InceptionB, self).__init__()
        self.branch1_1x1 = Basic_Conv(input_channel, channel_7x7, kernel_size = 1)
        self.branch1_1x7_1 = Basic_Conv(channel_7x7, channel_7x7, kernel_size = (1, 7), padding = (0, 3))
        self.branch1_7x1_1 = Basic_Conv(channel_7x7, channel_7x7, kernel_size = (7, 1), padding = (3, 0))
        self.branch1_1x7_2 = Basic_Conv(channel_7x7, channel_7x7, kernel_size = (1, 7), padding = (0, 3))
        self.branch1_7x1_2 = Basic_Conv(channel_7x7, 192, kernel_size = (7, 1), padding = (3, 0))
        
        self.branch2_1x1 = Basic_Conv(input_channel, channel_7x7, kernel_size = 1)
        self.branch2_1x7 = Basic_Conv(channel_7x7, channel_7x7, kernel_size = (1, 7), padding = (0, 3))
        self.branch2_7x1 = Basic_Conv(channel_7x7, 192, kernel_size = (7, 1), padding = (3, 0))
        
        self.branch3_pool = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.branch3_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        
        self.branch4_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        
    def forward(self, x):
        #Type : (Tensor)
        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_1x7_1(branch1)
        branch1 = self.branch1_7x1_1(branch1)
        branch1 = self.branch1_1x7_2(branch1)
        branch1 = self.branch1_7x1_2(branch1)
        
        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_1x7(branch2)
        branch2 = self.branch2_7x1(branch2)
        
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_1x1(branch3)
        
        branch4 = self.branch4_1x1(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim = 1)
        
class InceptionC(nn.Module):
    
    def __init__(self, input_channel):
        #Type : (int)
        super(InceptionC, self).__init__()
        self.branch1_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        self.branch1_3x3 = Basic_Conv(64, 92, kernel_size = 3, padding = 1)
        self.branch1_3x1 = Basic_Conv(92, 92, kernel_size = (3, 1), padding = (1, 0))
        self.branch1_1x3 = Basic_Conv(92, 92, kernel_size = (1, 3), padding = (0, 1))
        
        self.branch2_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        self.branch2_3x1 = Basic_Conv(64, 92, kernel_size = (3, 1), padding = (1, 0))
        self.branch2_1x3 = Basic_Conv(64, 92, kernel_size = (1, 3), padding = (0, 1))
        
        self.branch3_pool = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.branch3_1x1 = Basic_Conv(input_channel, 92, kernel_size = 1)
        
        self.branch4_1x1 = Basic_Conv(input_channel, 92, kernel_size = 1)
        
    def forward(self, x):
        #Type : (Tensor)
        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_3x3(branch1)
        branch1_split1 = self.branch1_3x1(branch1)
        branch1_split2 = self.branch1_1x3(branch1)
        
        branch2 = self.branch2_1x1(x)
        branch2_split1 = self.branch2_3x1(branch2)
        branch2_split2 = self.branch2_1x3(branch2)
        
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_1x1(branch3)
        
        branch4 = self.branch4_1x1(x)
        
        return torch.cat([branch1_split1, branch1_split2, branch2_split1, branch2_split2, branch3, branch4], dim = 1)