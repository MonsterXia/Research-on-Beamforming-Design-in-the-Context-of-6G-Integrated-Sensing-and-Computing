# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/3/16 19:07
# @Function:
import argparse
import os
import csv
import copy
import time
import json
import torch
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
from BFNN import BFNN
from BFTN import BFTN
from MyDataSet import MyDataSet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.font_manager import FontProperties

# Set the matlabPlot title's Chinese font
font = FontProperties(fname=r'C:/Users/monst/AppData/Local/Microsoft/Windows/Fonts/STSongti-SC-Regular.ttf', size=12)


class CustomLoss(nn.Module):
    def __init__(self, nt):
        super(CustomLoss, self).__init__()
        self.Nt = nt

    def forward(self, csi, phase, snr):
        # phase.shape = [batch_size, nt]
        # csi.shape = [batch_size, nt]
        # snr.shape = [batch_size, 1]

        phase_real = torch.cos(phase)
        phase_imag = torch.sin(phase)
        v_rf = torch.complex(phase_real, phase_imag).unsqueeze(1).to(torch.complex64).clone().detach()
        csi = csi.unsqueeze(1).transpose(2, 1).to(torch.complex64).clone().detach()
        # v_rf.shape = [batch_size, 1, nt]
        # csi.shape = [batch_size, nt, 1]

        temp = torch.bmm(v_rf, csi).squeeze(2)
        # temp.shape = [batch_size, 1]

        capacity = torch.log(1 + snr / self.Nt * (torch.abs(temp) ** 2)) / torch.log(torch.tensor(2.0))
        # capacity.shape = [batch_size, 1]
        capacity_average = -capacity.sum() / capacity.size(0)
        # Get average capacity for every batch (scalar)

        return capacity_average


def get_loss_function(nt, device):
    criterion = CustomLoss(nt)
    criterion = criterion.to(device)

    return criterion


def train(model, train_loader, criterion, optimizer, scheduler, env, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, ncols=128)
    for i, (csi, h, snr) in enumerate(progress_bar):
        csi = csi.float().requires_grad_(True).to(device)
        h = h.float().requires_grad_(True).to(device)

        if env.name == 'n20db' or env.name == 'p0db' or env.name == 'p20db':
            snr = torch.pow(10, torch.ones([h.shape[0], 1]).to(device) * env.value / 10)
        snr = snr.float().requires_grad_(True).to(device)

        optimizer.zero_grad()

        outputs = model(h)

        loss = criterion(csi, outputs, snr)
        running_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()

        progress_bar.update()
        progress_bar.set_postfix({"train_loss": loss.item()})
    progress_bar.close()

    return running_loss / len(train_loader)


def test(model, train_loader, criterion, env, device):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, ncols=128)
    for i, (csi, h, snr) in enumerate(progress_bar):
        csi = csi.float().requires_grad_(True).to(device)
        h = h.float().requires_grad_(True).to(device)

        if env.name == 'n20db' or env.name == 'p0db' or env.name == 'p20db':
            snr = torch.pow(10, torch.ones([h.shape[0], 1]).to(device) * env.value / 10)
        snr = snr.float().requires_grad_(True).to(device)

        outputs = model(h)

        loss = criterion(csi, outputs, snr)
        running_loss += loss.item()

        progress_bar.update()
        progress_bar.set_postfix({"test_loss": loss.item()})
    progress_bar.close()

    return -running_loss / len(train_loader)


def vali(model, train_loader, criterion, device):
    model.eval()

    capacity = []

    for snr in range(-20, 25, 5):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, ncols=128)
        for i, (csi_, h, _) in enumerate(progress_bar):
            csi = csi_.float().requires_grad_(True).to(device)
            h = h.float().requires_grad_(True).to(device)

            noise = torch.pow(10, torch.ones([h.shape[0], 1]).to(device) * snr / 10)
            noise = noise.float().requires_grad_(True).to(device)

            outputs = model(h)

            loss = criterion(csi, outputs, noise)
            running_loss += loss.item()

            progress_bar.update()
            progress_bar.set_postfix({"val_loss": loss.item()})
        progress_bar.close()

        temp = -running_loss / len(train_loader)

        capacity.append(temp)
    return capacity


def dataset_path_append(dataset_root, condition):
    dataset_path = ""
    if condition == 'n20db':
        dataset_path = os.path.join(dataset_root, '-20db')
    elif condition == 'p0db':
        dataset_path = os.path.join(dataset_root, '0db')
    elif condition == 'p20db':
        dataset_path = os.path.join(dataset_root, '20db')
    elif condition == 'Lest1':
        dataset_path = os.path.join(dataset_root, 'Lest1')
    elif condition == 'Lest2':
        dataset_path = os.path.join(dataset_root, 'Lest2')
    elif condition == 'Lest3':
        dataset_path = os.path.join(dataset_root, '20db')

    return dataset_path


def get_data(dataset_root, batch_size):
    dataset_train = MyDataSet(dataset_path=os.path.join(dataset_root, 'train'))
    dataset_test = MyDataSet(dataset_path=os.path.join(dataset_root, 'test'))

    train_dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def get_model(model_mode, model_root, condition, nt, device):
    model_name = f"{model_mode}_{condition}.pth"
    model_path = os.path.join(model_root, model_mode, model_name)
    logs_out = []
    if os.path.isfile(model_path):
        logs_out.append(f"Model {model_mode}_{condition} exist, try to load model! {get_time()}")
        model = torch.load(model_path)
        logs_out.append(f"Load model successfully! {get_time()}")
    else:
        if model_mode == 'BFNN':
            model = BFNN(nt)
        elif model_mode == 'BFTN':
            model = BFTN(nt)
        logs_out.append(f"Create model successfully! {get_time()}")
        torch.save(model, model_path)
    model = model.to(device)

    return model, logs_out


def update_best_model(model, capacity):
    best_model = copy.deepcopy(model)
    best_capacity = capacity

    return best_model, best_capacity


def write_to_csv(csv_root, model_mode, condition, data):
    csv_name = f"{model_mode}_capacity.csv"
    csv_path = os.path.join(csv_root, csv_name)
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([f"{model_mode}_{condition}"])
        writer.writerows([data])


def read_from_csv(csv_root, model_mode):
    csv_name = f"{model_mode}_capacity.csv"
    csv_path = os.path.join(csv_root, csv_name)
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def write2json(json_root, data_list):
    current_time = get_time()
    filename_time = current_time.replace(" ", "_").replace(":", "_")

    # 将数据列表和时间合并为一个字典
    data_to_save = {
        "timestamp": current_time,
        "output_list": data_list
    }

    # 将数据转换为JSON字符串
    json_data = json.dumps(data_to_save, indent=4)

    # 指定保存路径
    path = f"{json_root}/train_records_{filename_time}.json"

    # 将JSON字符串写入文件
    with open(path, 'w', encoding="utf-8") as file:
        file.write(json_data)

    print(f"Data saved to {path}")


def plot_data_chinese(data, title, fig_root, model_mode):
    names = []
    values = []

    # 3 sets of data
    for i in range(0, 6, 2):
        names.append(data[i][0])
        values.append(list(map(float, data[i + 1])))

    # x's range
    db_range = np.arange(-20, 25, 5)

    # new fig
    plt.figure()
    for name, value in zip(names, values):
        plt.plot(db_range, value, marker='o', label=name)

    plt.xlabel('信噪比 （dB）', fontproperties=font)
    plt.ylabel('频谱效率（（bit/s）/Hz）', fontproperties=font)
    plt.title(f'不同{title}下{model_mode}的频谱效率与环境信噪比的关系', fontproperties=font)
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(fig_root, f'{model_mode}_{title}')
    plt.savefig(file_path)

    plt.show()


def plot_data(data, title, fig_root, model_mode):
    names = []
    values = []

    # 3 sets of data
    for i in range(0, 6, 2):
        names.append(data[i][0])
        values.append(list(map(float, data[i + 1])))

    # x's range
    db_range = np.arange(-20, 25, 5)

    # new fig
    plt.figure()
    for name, value in zip(names, values):
        plt.plot(db_range, value, marker='o', label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectrum Effectiveness( (bit/s)/Hz )')
    plt.title(f'{model_mode}‘s Spectrum Effectiveness vs. {title}')
    plt.grid(True)
    plt.legend()

    file_path = os.path.join(fig_root, f'{model_mode}_{title}')
    plt.savefig(file_path)

    plt.show()


def get_model_structure(model):
    structure = []
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        structure.append([name, num_params])

    return structure


def disable_warning():
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="Casting complex values to real discards the imaginary part")


def parse_args():
    parser = argparse.ArgumentParser(description='Run validation with specified model mode.')
    parser.add_argument('--model_mode', type=str, default="BFNN", help='The mode of the model etc.BFNN/BFTN')
    parser.add_argument('--epochs', type=int, default=200, help='Rounds of training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    return parser.parse_args()
