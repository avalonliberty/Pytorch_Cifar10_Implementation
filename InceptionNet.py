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
        self.branch1_7x1_2 = Basic_Conv(channel_7x7, 196, kernel_size = (7, 1), padding = (3, 0))
        
        self.branch2_1x1 = Basic_Conv(input_channel, channel_7x7, kernel_size = 1)
        self.branch2_1x7 = Basic_Conv(channel_7x7, channel_7x7, kernel_size = (1, 7), padding = (0, 3))
        self.branch2_7x1 = Basic_Conv(channel_7x7, 196, kernel_size = (7, 1), padding = (3, 0))
        
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
        self.branch1_3x3 = Basic_Conv(64, 96, kernel_size = 3, padding = 1)
        self.branch1_3x1 = Basic_Conv(96, 96, kernel_size = (3, 1), padding = (1, 0))
        self.branch1_1x3 = Basic_Conv(96, 96, kernel_size = (1, 3), padding = (0, 1))
        
        self.branch2_1x1 = Basic_Conv(input_channel, 64, kernel_size = 1)
        self.branch2_3x1 = Basic_Conv(64, 96, kernel_size = (3, 1), padding = (1, 0))
        self.branch2_1x3 = Basic_Conv(64, 96, kernel_size = (1, 3), padding = (0, 1))
        
        self.branch3_pool = nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.branch3_1x1 = Basic_Conv(input_channel, 96, kernel_size = 1)
        
        self.branch4_1x1 = Basic_Conv(input_channel, 96, kernel_size = 1)
        
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
    
class InceptionD(nn.Module):
    
    def __init__(self, input_channel):
        #Type : (int)
        super(InceptionD, self).__init__()
        self.branch1_1x1 = nn.Conv2d(input_channel, 64, kernel_size = 1)
        self.branch1_3x3_1 = nn.Conv2d(64, 96, kernel_size = 3, padding = 1)
        self.branch1_3x3_2 = nn.Conv2d(96, 96, kernel_size = 3, stride = 2, padding = 1)
        
        self.branch2_1x1 = nn.Conv2d(input_channel, 64, kernel_size = 1)
        self.branch2_3x3 = nn.Conv2d(64, 96, kernel_size = 3, stride = 2, padding = 1)
        
        self.branch3 = nn.AvgPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        #Type : (Tensor)
        branch1 = self.branch1_1x1(x)
        branch1 = self.branch1_3x3_1(branch1)
        branch1 = self.branch1_3x3_2(branch1)
        
        branch2 = self.branch2_1x1(x)
        branch2 = self.branch2_3x3(branch2)
        
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], dim = 1)
    
class Inception_AUX(nn.Module):
    
    def __init__(self):
        super(Inception_AUX, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv1 = Basic_Conv(520, 128, kernel_size = 1)
        self.conv2 = Basic_Conv(128, 768, kernel_size = 3, stride = 2, padding = 1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, 10)
        
    def forward(self, x):
        #Type : (Tensor)
        # N x 520 x 8 x 8
        aux = self.avg_pool(x)
        # N x 520 x 4 x 4
        aux = self.conv1(aux)
        # N x 128 x 4 x 4
        aux = self.conv2(aux)
        # N x 768 x 4 x 4
        aux = self.avg(aux)
        # N x 768 x 1 x 1
        aux = torch.flatten(aux, 1)
        # N x 768
        aux = self.fc(aux)
        # N x 10
        
        return aux
    
class InceptionV3(nn.Module):
    
    def __init__(self, aux_logit = True):
        super(InceptionV3, self).__init__()
        self.aux_logit = aux_logit and self.training
        
        self.pre_conv1 = Basic_Conv(3, 32, kernel_size = 3, stride = 2, padding = 1)
        self.pre_conv2 = Basic_Conv(32, 32, kernel_size = 1)
        self.IncepA1 = InceptionA(32, 32)
        self.IncepA2 = InceptionA(256, 64)
        self.IncepA3 = InceptionA(288, 64)
        self.reduction1 = InceptionD(288)
        self.IncepB1 = InceptionB(480, 128)
        self.IncepB2 = InceptionB(520, 160)
        self.IncepB3 = InceptionB(520, 160)
        self.IncepB4 = InceptionB(520, 192)
        self.aux = Inception_AUX()
        self.reduction2 = InceptionD(520)
        self.IncepC1 = InceptionC(712)
        self.IncepC2 = InceptionC(576)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.5)
        self.fc = nn.Linear(576, 10)
        
    def forward(self, x):
        #Type : (Tensor)
        # N x 3 x 32 x 32
        x = self.pre_conv1(x)
        # N x 32 x 16 x 16
        x = self.pre_conv2(x)
        # N x 32 x 16 x 16
        x = self.IncepA1(x)
        # N x 256 x 16 x 16
        x = self.IncepA2(x)
        # N x 288 x 16 x 16
        x = self.IncepA3(x)
        # N x 288 x 16 x 16
        x = self.reduction1(x)
        # N x 480 x 8 x 8
        x = self.IncepB1(x)
        # N x 520 x 8 x 8
        x = self.IncepB2(x)
        # N x 520 x 8 x 8
        x = self.IncepB3(x)
        # N x 520 x 8 x 8
        x = self.IncepB4(x)
        # N x 520 x 8 x 8
        if self.aux_logit:
            aux = self.aux(x)
        else:
            aux = None
        # N x 520 x 8 x 8
        x = self.reduction2(x)
        # N x 712 x 4 x 4
        x = self.IncepC1(x)
        # N x 576 x 4 x 4
        x = self.IncepC2(x)
        # N x 576 x 4 x 4
        x = self.avg(x)
        # N x 576 x 1 x 1
        x = torch.flatten(x , 1)
        # N x 576
        x = self.dropout(x)
        # N x 576
        x = self.fc(x)
        # N x 10
        return x, aux