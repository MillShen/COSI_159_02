import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularPenalty(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-7, m=None):
        super(AngularPenalty, self).__init__()

        self.m = 4 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.f = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.f.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)
        wf = self.f(x)

        num = torch.cos(self.m * torch.acos(
            torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1 + self.eps, 1-self.eps)))
        e = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        den = torch.exp(num) + torch.sum(torch.exp(e), dim=1)
        L = num - torch.log(den)
        return -torch.mean(L)

class SphereFace(nn.Module):
    def __init__(self, class_num: int, features=False):
        super(SphereFace, self).__init__()
        self.class_num = class_num
        self.features = features

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=138, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.fc = nn.Linear(512 * 5 * 5, 512)
        self.angular_margin = AngularPenalty(512, self.class_num)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.features:
            return x
        else:
            return self.angular_margin(x, y)

