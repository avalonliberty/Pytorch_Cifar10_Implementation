import torch.nn as nn
from ResNet import *
from segmentation.FeaturePyramidNetwork import FeaturePyramidNetwork
from utils import get_intermediate_layers

class backbone_with_FPN(nn.Module):
    
    def __init__(self, backbone, return_layers, input_channels, out_channel):
        super(backbone_with_FPN, self).__init__()
        self.gl = get_intermediate_layers(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(input_channels, out_channel)
        
    def forward(self, x):
        ft = self.gl(x)
        output = self.fpn(ft)
        return output
    
def FPN_with_ResNet(backbone_name):
    pred_name = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101']
    assert backbone_name in pred_name, f'backbone should be {" or ".join(pred_name)}'
    backbone = eval(backbone_name)()
    return_layers = {'layer1' : 'feat1',
                     'layer2' : 'feat2',
                     'layer3' : 'feat3',
                     'layer4' : 'feat4'}
    input_channels= [backbone.origin_input,
                     backbone.origin_input * 2,
                     backbone.origin_input * 4,
                     backbone.origin_input * 8]
    out_channel = 256
    fpn = backbone_with_FPN(backbone, return_layers, input_channels, out_channel)
    return fpn