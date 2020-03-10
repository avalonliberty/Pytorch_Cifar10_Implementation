from collections import OrderedDict
import torch.nn.functional as F
from torch import nn

class FeaturePyramidNetwork(nn.Module):
    
    def __init__(self, input_channels, output_channel):
        super(FeaturePyramidNetwork, self).__init__()
        self.InnerConvLayer = nn.ModuleList()
        self.OuterConvLayer = nn.ModuleList()
        for input_channel in input_channels:
            InnerConv = nn.Conv2d(input_channel, output_channel, 1)
            OuterConv = nn.Conv2d(output_channel, output_channel, 3, padding = 1)
            self.InnerConvLayer.append(InnerConv)
            self.OuterConvLayer.append(OuterConv)
            
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a = 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        result = []
        
        last_layer = self.InnerConvLayer[-1](x[-1])
        last_layer = self.OuterConvLayer[-1](last_layer)
        result.append(last_layer)
        
        for index in range(len(x) - 2, -1, -1):
            bottom_top = self.InnerConvLayer[index](x[index])
            lateral_size = bottom_top.size()[-2:]
            top_down = F.interpolate(last_layer, lateral_size, mode = 'nearest')
            last_layer = bottom_top + top_down
            result.insert(0, self.OuterConvLayer[index](last_layer))
        
        output = OrderedDict([(name, tensor) for name, tensor in zip(names, result)])
        
        return output
                
        
