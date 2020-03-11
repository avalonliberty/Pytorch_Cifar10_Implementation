'''
Pytorch implementation of ResNet
Deep Residual Learning for Image Recognition<https://arxiv.org/abs/1512.03385>
'''

import torch
import torch.nn as nn

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
        self.relu = nn.ReLU()
        if stride != 1:
            self.downsampling = BasicConv(input_channels,
                                          output_channels,
                                          kernel_size = 1,
                                          stride = 2,
                                          padding = 0)
        else:
            self.downsampling = None
        
    def forward(self, x):
        identity = x
        if self.downsampling is not None:
            identity = self.downsampling(x)
        x = self.comp1(x)
        x = self.comp2(x)
        x = x + identity
        x = self.relu(x)
        return x
    
class BottleNeck(nn.Module):
    
    expansion = 4
    
    def __init__(self, input_channels, output_channels, stride, groups = 1):
        super(BottleNeck, self).__init__()
        inner_width = int(output_channels / self.expansion)
        self.reduce = BasicConv(input_channels, inner_width, kernel_size = 1)
        self.conv = BasicConv(inner_width, inner_width, kernel_size = 3,
                              stride = stride, groups = groups, padding = 1)
        self.restore = BasicConv(inner_width, output_channels, kernel_size = 1)
        if stride != 1:
            self.downsampling = BasicConv(input_channels,
                                          output_channels,
                                          kernel_size = 1,
                                          stride = 2,
                                          padding = 0)
        else:
            self.downsampling = None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        if self.downsampling is not None:
            identity = self.downsampling(x)
        x = self.reduce(x)
        x = self.conv(x)
        x = self.restore(x)
        x = identity + x
        x = self.relu(x)
        return x
    
class flatten(nn.Module):
    
    def __init__(self):
        super(flatten, self).__init__()
        
    def forward(self, x):
        return torch.flatten(x, 1)
    
class ResNet(nn.Module):
    
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.origin_input = 64
        self.input_channels = self.origin_input
        self.preconv = BasicConv(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.layer1 = self._create_layers(block, layers[0], 64, 1)
        self.layer2 = self._create_layers(block, layers[1], 128, 2)
        self.layer3 = self._create_layers(block, layers[2], 256, 2)
        self.layer4 = self._create_layers(block, layers[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = flatten()
        self.fc = nn.Linear(512, 10)
        
    def _create_layers(self, block, layer, output_channels, stride):
        strides = [stride] + [1] * (layer - 1)
        layer_holder = []
        for stride in strides:
            layer_holder.append(block(self.input_channels,
                                  output_channels,
                                  stride = stride))
            self.input_channels = output_channels
        return nn.Sequential(*layer_holder)
    
    def forward(self, x):
        x = self.preconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
def ResNet18():
    
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    
    return ResNet(BottleNeck, [3, 4, 24, 3])