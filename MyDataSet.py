# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/17 0:25
# @Function:
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, dataset_path):
        # load the perfect csi
        data_h = sio.loadmat(dataset_path + '/pcsi.mat')['pcsi']
        data_h = np.squeeze(data_h)
        data_h_tensor = torch.from_numpy(data_h)
        self.h = data_h_tensor

        # load the estimated csi
        data_h_est = sio.loadmat(dataset_path + '/ecsi.mat')['ecsi']
        data_h_est = np.expand_dims(np.concatenate([np.real(data_h_est), np.imag(data_h_est)], 1), 1)
        data_h_est_tensor = torch.from_numpy(data_h_est)
        self.h_est = data_h_est_tensor

    def __len__(self):
        return len(self.h)

    def __getitem__(self, index):
        snr = np.power(10, np.random.randint(-20, 20, 1) / 10)
        snr_tensor = torch.from_numpy(snr)
        return self.h[index], self.h_est[index], snr_tensor




