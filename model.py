from torch import nn
import torch

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.layer1 = nn.Linear(14,30)
        self.layer2 = nn.Linear(30,5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        return out2

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.layer1 = nn.Linear(14,20)
        self.layer2 = nn.Linear(20,30)
        self.layer3 = nn.Linear(30, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        return out3

class Classifier3(nn.Module):
    def __init__(self):
        super(Classifier3, self).__init__()
        self.layer1 = nn.Linear(14,40)
        self.layer2 = nn.Linear(40,100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 20)
        self.layer5 = nn.Linear(20, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        out4 = self.act(self.layer4(out3))
        out5 = self.act(self.layer5(out4))
        return out5

class Classifier4(nn.Module):
    def __init__(self):
        super(Classifier4, self).__init__()
        self.layer1 = nn.Linear(14,17)
        self.layer2 = nn.Linear(17,8)
        self.layer3 = nn.Linear(8, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        out3 = self.act(self.layer3(out2))
        return out3