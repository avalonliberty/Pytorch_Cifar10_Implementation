'''
Pytorch implementation of ResNet
Deep Residual Learning for Image Recognition<https://arxiv.org/abs/1512.03385>
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size = kernel_size,
                              bias = False,
                              **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    
class BasicBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, stride):
        super(BasicBlock, self).__init__()
        self.downsampling = nn.Sequential()
        self.comp1 = BasicConv(input_channels,
                               output_channels,
                               kernel_size = 3,
                               stride = stride,
                               padding = 1)
        self.comp2 = BasicConv(output_channels,
                               output_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        if stride != 1:
            self.downsampling = BasicConv(input_channels,
                                          output_channels,
                                          kernel_size = 1,
                                          stride = 2,
                                          padding = 0)
        
    def forward(self, x):
        identity = self.downsampling(x)
        x = self.comp1(x)
        x = self.comp2(x)
        x += identity
        x = F.relu(x)
        return x
    
class BottleNeck(nn.Module):
    
    expansion = 4
    def __init__(self, input_channels, stride):
        super(BottleNeck, self).__init__()
        self.reduce = BasicConv(input_channels)
    
class ResNet(nn.Module):
    
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.input_channels = 64
        self.layer1 = self._create_layers(block, layers[0], 64, 1)
        self.layer2 = self._create_layers(block, layers[1], 128, 2)
        self.layer3 = self._create_layers(block, layers[2], 256, 2)
        self.layer4 = self._create_layers(block, layers[3], 512, 2) 
        
    def _create_layers(self, block, layer, output_channels, stride):
        strides = [stride] + [1] * (layer - 1)
        layer_holder = []
        for stride in strides:
            layer_holder.append(block(self.input_channels,
                                  output_channels,
                                  stride = stride))
            self.input_channels = output_channels
        return nn.Sequential(*layer_holder)