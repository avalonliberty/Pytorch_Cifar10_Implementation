'''
Implementation of Densenet
Densely Connected Convolutional Networks<https://arxiv.org/abs/1608.06993>
'''
import torch.nn as nn
import torch

class dense_layer(nn.Module):
    
    def __init__(self, input_channel, growth_rate):
        #Type : (int, int)
        super(dense_layer, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(input_channel, 4 * growth_rate, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv_3x3 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size = 3, padding  = 1, bias = False)
        
    def forward(self, x):
        #Type : (Tennsor)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_1x1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv_3x3(x)
        return x
    
class dense_block(nn.Module):
    
    def __init__(self, input_channel, layers, growth_rate):
        #Type : (int, int, int)
        super(dense_block, self).__init__()
        self.layers = nn.ModuleList([dense_layer(input_channel + layer * growth_rate, growth_rate) for layer in range(layers)])
        
    def forward(self, x):
        #Type : (Tensor)
        features = [x]
        for index in range(len(self.layers)):
            feature = torch.cat(features, dim = 1)
            new_feature = self.layers[index](feature)
            features.append(new_feature)
        return torch.cat(features, 1)
    
class transition_block(nn.Module):
    
    def __init__(self, input_channel, output_channel):
        #Type : (int, int)
        super(transition_block, self).__init__()
        self.bn = nn.BatchNorm2d(input_channel)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size = 1, bias = False)
        self.avg = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        #Type : (Tensor)
        x = self.bn(x)
        x = self.conv(x)
        x = self.avg(x)
        return x
    
class flatten(nn.Module):
    
    def __init__(self):
        super(flatten, self).__init__()
        
    def forward(self, x):
        #Type : (Tensor)
        return torch.flatten(x, 1)
    
class denseNet(nn.Module):
    
    def __init__(self, cfg, input_channel = 64, growth_rate = 32):
        #Type : (int, List(int * 4))
        super(denseNet, self).__init__()
        self.pre_layer = nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace = True)
                )
        layer = []
        num_feature = input_channel
        for index, num_layer in enumerate(cfg):
            layer.append(dense_block(num_feature, num_layer, growth_rate))
            num_feature = num_feature + num_layer * growth_rate
            if index != len(cfg) - 1:
                layer.append(transition_block(num_feature, num_feature // 2))
                num_feature = num_feature // 2
        self.features = nn.Sequential(*layer)
        self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                flatten(),
                nn.Dropout(p = 0.5),
                nn.Linear(num_feature, 10)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        
    def forward(self, x):
        #Type : (Tensor)
        # N x 3 x 32 x 32
        x = self.pre_layer(x)
        # N x 32 x 16 x 16
        x = self.features(x)
        x = self.classifier(x)
        
        return x
    
def denseNet121():
    
    return denseNet([6, 12, 24, 16])

def denseNet161():
    
    return denseNet([6, 12, 36, 24], input_channel = 96, growth_rate = 48)

def denseNet169():
    
    return denseNet([6, 12, 32, 32])

def denseNet201():
    
    return denseNet([6, 12, 48, 32])