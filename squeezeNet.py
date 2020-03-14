'''
Implementation of squeezenet
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size<https://arxiv.org/abs/1602.07360>
'''

import torch.nn as nn
import torch
import torch.nn.init as init
from collections import OrderedDict

class squeeze_block(nn.Module):
    
    def __init__(self, input_channel, squeeze_channel):
        #type : (int, int)
        super(squeeze_block, self).__init__()
        self.squeeze = nn.Conv2d(input_channel, squeeze_channel, kernel_size = 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.squeeze(x))

class expand_block(nn.Module):
    
    def __init__(self, input_channel, channel_x1, channel_x3):
        #type : (int, int, int)
        super(expand_block, self).__init__()
        self.expand_x1 = nn.Sequential(nn.Conv2d(input_channel, channel_x1, kernel_size = 1),
                                 nn.ReLU(inplace = True))
        self.expand_x3 = nn.Sequential(nn.Conv2d(input_channel, channel_x3, kernel_size = 3, padding = 1),
                                 nn.ReLU(inplace = True))
        
    def forward(self, x):
        output_x1 = self.expand_x1(x)
        output_x3 = self.expand_x3(x)
        output = torch.cat([output_x1, output_x3], dim = 1)
        return output
    
class fire(nn.Module):
    
    def __init__(self, input_channel, squeeze_channel, channel_x1, channel_x3):
        #type : (int, int, int, int)
        super(fire, self).__init__()
        self.squeeze = squeeze_block(input_channel, squeeze_channel)
        self.expand = expand_block(squeeze_channel, channel_x1, channel_x3)
        
    def forward(self, x):
        x = self.squeeze(x)
        x = self.expand(x)
        return x
    
class squeezeNet(nn.Module):
    
    def __init__(self, num_classes = 10):
        #type : (int)
        super(squeezeNet, self).__init__()
        self.prev_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True)
                )
        self.fire = nn.Sequential(
                fire(64, 16, 64, 64),
                fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
                fire(128, 32, 128, 128),
                fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size = 3, stride = 2, ceil_mode = True),
                fire(256, 48, 192, 192),
                fire(384, 48, 192, 192),
                fire(384, 48, 256, 256),
                fire(512, 48, 256, 256)
                )
        self.post_layer = nn.Sequential(
                OrderedDict([
                        ('dropout', nn.Dropout(p = 0.5)),
                        ('last_conv', nn.Conv2d(512, num_classes, kernel_size = 1)),
                        ('relu', nn.ReLU(inplace = True)),
                        ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
                        ])
                )
    
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if name == 'post_layer.last_conv':
                    init.normal_(module.weight, mean = 0, std = 0.01)
                else:
                    init.kaiming_normal_(module.weight)
                init.constant_(module.bias, 0)
        
    def forward(self, x):
        #type : (Tensor)
        x = self.prev_layer(x)
        x = self.fire(x)
        x = self.post_layer(x)
        x = torch.flatten(x, 1)
        return x