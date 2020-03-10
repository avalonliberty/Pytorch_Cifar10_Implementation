import torch.nn as nn
from collections import OrderedDict

class get_intermediate_layers(nn.Module):
    
    def __init__(self, backbone, return_layers):
        super(get_intermediate_layers, self).__init__()
        self.backbone = backbone
        self.return_layers = return_layers
        
    def forward(self, x):
        module_list = OrderedDict()
        output = {}
        for name, module in self.backbone.named_children():
            module_list[name] = module
        for name, module in module_list.items():
            x = module(x)
            if name in self.return_layers:
                output[self.return_layers[name]] = x
        return output