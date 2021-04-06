from torch import nn
import torch

class QualityClassifier(nn.Module):
    def __init__(self):
        super(QualityClassifier,self).__init__()
        self.layer1 = nn.Linear(14,30)
        self.layer2 = nn.Linear(30,5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out1 = self.act(self.layer1(x))
        out2 = self.act(self.layer2(out1))
        return out2