# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/23 17:12
# @Function:
import torch.nn as nn


# Define the BFNN model
class BFTN(nn.Module):
    def __init__(self, nt):
        super(BFTN, self).__init__()
        self.dropout_layer = nn.Dropout(p=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.normalization_layer = nn.BatchNorm1d(2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, norm=self.normalization_layer,
                                                         num_layers=6)
        self.flatten_layer = nn.Flatten()
        self.linear_layer = nn.Linear(2 * nt, nt)

    def forward(self, x):
        x = self.dropout_layer(x.squeeze())
        x = self.transformer_encoder(x)
        x = self.flatten_layer(x)
        phase = self.linear_layer(x)
        return phase
