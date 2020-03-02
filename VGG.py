#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:29:27 2020

@author: avalon
"""

import torch.nn as nn

class VGG(nn.Module):
    '''
    This class is designed to implement various VGG structures
    '''
    
    def __init__(self, variation, is_norm = True):
        super(VGG, self).__init__()
        self.is_norm = is_norm
        self.variations = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
        self.cfgs = {'VGG11' : [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
                     'VGG13' : [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
                     'VGG16' : [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
                     'VGG19' : [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 256, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']}
        assert(variation.upper() in self.variations)
        self.cfg = self.cfgs[variation]
        self.backbone = self._create_layers()
        self.fc = nn.Sequential(nn.Dropout(p = 0.5),
                                nn.Linear(512, 1024),
                                nn.ReLU(inplace = True),
                                nn.Dropout(p = 0.5),
                                nn.Linear(1024, 128),
                                nn.ReLU(inplace = True),
                                nn.Linear(128, 10))
        
    def _create_layers(self):
        input_channels = 3
        module_holder = []
        for para in self.cfg:
            if para == 'P':
                module_holder.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            else:
                conv = nn.Conv2d(in_channels = input_channels,
                                 out_channels = para,
                                 kernel_size = 3,
                                 padding = 1)
                if self.is_norm:
                    module_holder += [conv, nn.BatchNorm2d(para), nn.ReLU(inplace = True)]
                else:
                    module_holder += [conv, nn.ReLU(inplace = True)]
        return nn.Sequential(*module_holder)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 512)
        output = self.fc(x)
        return output
                
        
model = VGG('VGG19')

        