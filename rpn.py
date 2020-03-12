import torch.nn as nn

class RPNhead(nn.Module):
    
    def __init__(self, input_channels, num_anchors):
        super(RPNhead, self).__init__()
        self.preprocessing = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True)
                )
        self.cls = nn.Conv2d(input_channels, num_anchors * 2, kernel_size = 1)
        self.regre = nn.Conv2d(input_channels, num_anchors * 4, kernel_size = 1)
        
    def forward(self, x):
        #type : (List[Tensor])
        cls = []
        regre = []
        for ft in x:
            x = self.preprocessing(x)
            cls.append(self.cls(x))
            regre.append(self.regre(x))
        return cls, regre