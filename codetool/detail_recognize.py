import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def get_project_root(file_name: str) -> str:
    upper_path = os.path.dirname(file_name)
    if 'visual.py' in os.listdir(upper_path):
        return upper_path
    else:
        return get_project_root(upper_path)


project_root = get_project_root(os.path.abspath(__file__))
script_path = os.path.abspath(__file__)
# inputFilePath = f'{project_root}/dataset/Restores/E3DTV_Ft_Myers_data_35.mat'
# inputFilePath = f'{project_root}/dataset/AVIRIS/test/hsi/California1_data_71.mat'
# inputFilePath = f'{project_root}/dataset/AVIRIS/test/hsi/Ft_Myers_data_35.mat'
inputFilePath = f'{project_root}/dataset/AVIRIS/test/hsi/HyspIRI_3_line17_data_279.mat'
# inputFilePath = f'{project_root}/dataset/AVIRIS/test/hsi/HyspIRI_3_line40_data_263.mat'
# inputFilePath = f'{project_root}/dataset/AVIRIS/train/hsi/Moffett10.mat'
# mat_data = loadmat(inputFilePath)
# data = mat_data[list(mat_data.keys())[-1]]

model_name = "DCSN"

fig_name = "HyspIRI_3_line17_data_279"
data = np.load(f"./npys/{model_name}_{fig_name}.npy")
# 指定 RGB 通道
band_set = [24, 14, 5]  # 对应 MATLAB 的 band_set

# 提取指定波段数据
temp_show = data[:, :, band_set]

# 标准化函数（与 MATLAB 代码中一致）
def norm_color(R):
    R_norm = (R - np.mean(R)) / np.std(R)
    R_clipped = np.clip(R_norm, -2, 2)
    return R_clipped / 3 + 0.5
# 对提取的波段数据进行标准化
temp_show = norm_color(temp_show)
# 显示图像
# plt.imshow(temp_show[0:50, 50:100, :])
# plt.imshow(temp_show[180:220, 200:240, :])
# plt.imshow(temp_show[80:120, 100:140, :])
# plt.imshow(temp_show[80:120, 80:120, :])  # HyspIRI_3_line40_data_263
plt.imshow(temp_show[50:90, 160:200, :])  # HyspIRI_3_line40_data_263
plt.axis('off')  # 隐藏坐标轴
plt.tight_layout()
plt.show()
