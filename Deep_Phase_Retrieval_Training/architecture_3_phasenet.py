import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, input_channel, output_size):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(input_channel, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)

        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)

        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv9 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, 3, padding=1)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, int(output_size))

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        x = self.pool(F.tanh(self.conv4(x)))
        x = F.tanh(self.conv5(x))
        x = self.pool(F.tanh(self.conv6(x)))
        x = F.tanh(self.conv7(x))
        x = self.pool(F.tanh(self.conv8(x)))
        x = F.tanh(self.conv9(x))
        x = self.pool(F.tanh(self.conv10(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


# net = Net(10)
# torchsummary.summary(net, (3, 64, 64))