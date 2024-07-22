# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/16 19:48
# @Function:
import torch
import torch.nn as nn


# Define the BFNN model
class BFNN(nn.Module):
    def __init__(self, nt):
        super(BFNN, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(2*nt)
        self.flatten = nn.Flatten()
        self.batch_norm2 = nn.BatchNorm1d(2*nt)
        self.dense1 = nn.Linear(2*nt, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, nt)

    def forward(self, x):

        x = self.flatten(x)
        x = self.batch_norm1(x)
        x = self.batch_norm2(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.batch_norm3(x)
        x = self.dense2(x)
        x = torch.relu(x)
        phase = self.dense3(x)

        return phase

