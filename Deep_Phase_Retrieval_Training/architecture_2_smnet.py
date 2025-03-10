import torch
import torch.nn as nn
#import torch.nn.functional as F

def conv3x3(iChannels, oChannels, stride):
    return nn.Conv2d(iChannels, oChannels, (3, 3), (stride, stride), (1, 1))

def conv1x1(iChannels, oChannels, stride):
    return nn.Conv2d(iChannels, oChannels, (1, 1), (stride, stride), (0, 0))

class resBlock(nn.Module):
    def __init__(self, iChannels, oChannels, stride):
        super(resBlock, self).__init__()

        n = oChannels
        self.residual = nn.Sequential(
                conv3x3(int(iChannels), int(n/4), 1), nn.BatchNorm2d(int(n/4), 1e-3), nn.PReLU(),
                conv3x3(int(n/4), int(n/2), stride), nn.BatchNorm2d(int(n/2), 1e-3), nn.PReLU(),
                conv3x3(int(n/2), int(n), 1), nn.BatchNorm2d(int(n), 1e-3)
                )
        if stride > 1 or iChannels != n:
            self.identity = nn.Sequential(conv1x1(iChannels, oChannels, stride), nn.BatchNorm2d(int(n), 1e-3))
        else:
            self.identity = 'Identity'
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.residual(x)
        if self.identity == 'Identity':
            out += x
        else:
            out += self.identity(x)
        out = self.prelu(out)
        return out

class Net(nn.Module):
    def __init__(self, iChannels, output):
        super(Net, self).__init__()

        self.features = nn.Sequential(
                nn.Conv2d(int(iChannels), 64, 7, 1, 3), nn.BatchNorm2d(64, 1e-3), nn.PReLU(),
                nn.Conv2d(64, 128, 5, 1, 2), nn.BatchNorm2d(128, 1e-3), nn.PReLU(),
                resBlock(128, 128, 1), resBlock(128, 128, 1), resBlock(128, 128, 1),
                resBlock(128, 256, 4),
                resBlock(256, 256, 1), resBlock(256, 256, 1), resBlock(256, 256, 1),
                conv1x1(256, 128, 1), nn.BatchNorm2d(128, 1e-3), nn.PReLU(),
                conv1x1(128, 64,1), nn.BatchNorm2d(64, 1e-3), nn.PReLU(),
                conv1x1(64, 1, 1), nn.BatchNorm2d(1, 1e-3), nn.PReLU()
                )

        self.FCs = nn.Sequential(
            nn.Linear(256, 10),
            nn.PReLU(),
            nn.Linear(10, int(output))
        )
        self.Hardtanh = nn.Hardtanh(-20, 20)


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, x.size(2)*x.size(3))
        x = self.FCs(x)
        x = self.Hardtanh(x)
        return x


# model = Net(3, 10)
# torchsummary.summary(model, (3,64,64))