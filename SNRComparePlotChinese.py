# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2024/4/11 15:26
# @Function:

import csv

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# Set the matlabPlot title's Chinese font
font = FontProperties(fname=r'C:/Users/monst/AppData/Local/Microsoft/Windows/Fonts/STSongti-SC-Regular.ttf', size=12)

csv_root = "data"
fig_root = "Figs"

names = []
values = []

csv_path = r"data/SNRcompare.csv"
with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

db_range = np.arange(-20, 25, 5)


for i in range(0, 4, 2):
    names.append(data[i][0])
    values.append(list(map(float, data[i + 1])))

plt.figure()
for i, (name, value) in enumerate(zip(names, values)):
    marker = '^' if i == 0 else 'o'
    plt.plot(db_range, value, marker=marker, label=name)

plt.xlabel('信噪比 （dB）', fontproperties=font)
plt.ylabel('频谱效率（（bit/s）/Hz）', fontproperties=font)
plt.title('训练环境信噪比为0dB时2个模型的频谱效率与环境信噪比的关系', fontproperties=font)
plt.grid(True)
plt.legend()

file_path = "Figs/SNRCompare.png"
plt.savefig(file_path)

plt.show()



