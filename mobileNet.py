'''
Implementation of mobilenet
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications<https://arxiv.org/abs/1704.0486100>
'''
import torch.nn as nn
import torch

class dw_conv(nn.Module):
    
    def __init__(self, input_channel, output_channel, stride = 1, groups = 1):
        #Type : (int, int, int, int)
        super(dw_conv, self).__init__()
        self.group_conv = nn.Conv2d(input_channel, output_channel, kernel_size = 3, stride = stride, groups = groups, bias = False, padding = 1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #Type : (Tensor)
        x = self.group_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class pw_conv(nn.Module):
    
    def __init__(self, input_channel, output_channel):
        #Type : (int, int)
        super(pw_conv, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_channel, output_channel, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #Type : (Tensor)
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class basic_block(nn.Module):
    
    def __init__(self, input_channel, output_channel, stride, width_mult = 1):
        #Type : (int, int, int, int)
        super(basic_block, self).__init__()
        self.lateral = (stride == 1) and (input_channel == output_channel)
        new_input = int(input_channel * width_mult)
        new_output = int(output_channel * width_mult)
        self.dw = dw_conv(new_input, new_input, stride = stride, groups = new_input)
        self.pw = pw_conv(new_input, new_output)
        
    def forward(self, x):
        if self.lateral:
            x = x + self.pw(self.dw(x))
        else:
            x = self.dw(x)
            x = self.pw(x)
        return x
        
class mobileNet(nn.Module):
    
    def __init__(self, width_mult = 1):
        #Type : (int)
        super(mobileNet, self).__init__()
        self.pre_layer = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True))
        self.dw1 = basic_block(32, 64, 1, width_mult)
        self.dw2 = basic_block(64, 128, 2, width_mult)
        self.dw3 = basic_block(128, 128, 1, width_mult)
        self.dw4 = basic_block(128, 256, 2, width_mult)
        self.dw5 = basic_block(256, 256, 1, width_mult)
        self.dw5 = basic_block(256, 512, 1, width_mult)
        self.dw6 = basic_block(512, 512, 1, width_mult)
        self.dw7 = basic_block(512, 512, 1, width_mult)
        self.dw8 = basic_block(512, 512, 1, width_mult)
        self.dw9 = basic_block(512, 512, 1, width_mult)
        self.dw10 = basic_block(512, 512, 1, width_mult)
        self.dw11 = basic_block(512, 1024, 2, width_mult)
        self.dw12 = basic_block(1024, 1024, 1, width_mult)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(1024, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # N x 3 x 32 x 32
        x = self.pre_layer(x)
        # N x 32 x 16 x 16
        x = self.dw1(x)
        # N x 64 x 16 x16
        x = self.dw2(x)
        # N x 128 x 8 x 8
        x = self.dw3(x)
        # N x 128 x 8 x 8
        x = self.dw4(x)
        # N x 256 x 4 x 4
        x = self.dw5(x)
        # N x 256 x 4 x 4
        x = self.dw6(x)
        # N x 512 x 4 x 4
        x = self.dw7(x)
        # N x 512 x 4 x 4
        x = self.dw8(x)
        # N x 512 x 4 x 4
        x = self.dw9(x)
        # N x 512 x 4 x 4
        x = self.dw10(x)
        # N x 512 x 4 x 4
        x = self.dw11(x)
        # N x 1024 x 2 x 2
        x = self.dw12(x)
        # N x 1024 x 2 x 2
        x = self.avg(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc(x)
        return x