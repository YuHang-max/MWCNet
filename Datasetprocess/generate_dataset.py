import os
from numpy import random
import time


def get_filenames(directory: str) -> list:
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames


def get_train_data(file_list: list) -> tuple:
    current_time = time.time()
    random.seed(int(current_time))
    random.shuffle(file_list)
    num_files = len(file_list)
    return file_list[:num_files // 10], file_list[num_files // 10:]

# 目标路径
target_path1 = './dataset/AVIRIS/train/hsi'
target_path2 = './dataset/AVIRIS/test/hsi'

# 调用函数并打印结果
file_list1 = get_filenames(target_path1)
file_list2 = get_filenames(target_path2)
with open('../trainpath/train.txt', mode='r', encoding='utf-8') as file:
    i = 0
    for line in file:
        i += 1
        if 'Moffett' in line:
            print(i, True)
print(len(file_list1))
print(len(file_list2))
file_list = file_list1 + file_list2
print(len(file_list))
print(file_list[0])
random.shuffle(file_list)
print(file_list[0])
california_dataset = []
hyspiri_17_dataset = []
hyspiri_40_dataset = []
ftmyers_dataset = []
amikougami_dataset = []
arizona_dataset = []
campbellriver_dataset = []
dewar_dataset = []
hama_dataset = []
lethbridge_dataset = []
qccut_dataset = []
seney_dataset = []
yellowstone_dataset = []
for filename in file_list:
    if 'California' in filename:
        california_dataset.append(filename)
    if 'Ft_Myers' in filename:
        ftmyers_dataset.append(filename)
    if 'line17' in filename:
        hyspiri_17_dataset.append(filename)
    if 'line40' in filename:
        hyspiri_40_dataset.append(filename)
    if 'Amikougami' in filename:
        amikougami_dataset.append(filename)
    if 'Arizona' in filename:
        arizona_dataset.append(filename)
    if 'Campbell' in filename:
        campbellriver_dataset.append(filename)
    if 'Dewar' in filename:
        dewar_dataset.append(filename)
    if 'Lethbridge' in filename:
        lethbridge_dataset.append(filename)
    if 'QC' in filename:
        qccut_dataset.append(filename)
    if 'Seney' in filename:
        seney_dataset.append(filename)
    if 'Yellow' in filename:
        yellowstone_dataset.append(filename)
print(len(california_dataset))
print(len(ftmyers_dataset))
print(len(hyspiri_17_dataset))
print(len(hyspiri_40_dataset))
print(len(amikougami_dataset))
print(len(arizona_dataset))
print(len(campbellriver_dataset))
print(len(dewar_dataset))
print(len(lethbridge_dataset))
print(len(qccut_dataset))
print(len(seney_dataset))
print(len(yellowstone_dataset))
random.shuffle(california_dataset)
random.shuffle(ftmyers_dataset)
random.shuffle(hyspiri_17_dataset)
random.shuffle(hyspiri_40_dataset)
random.shuffle(amikougami_dataset)
random.shuffle(arizona_dataset)
random.shuffle(campbellriver_dataset)
random.shuffle(dewar_dataset)
random.shuffle(lethbridge_dataset)
random.shuffle(qccut_dataset)
random.shuffle(seney_dataset)
random.shuffle(yellowstone_dataset)
print(len(california_dataset))
valid_california_dataset, train_califirnia_dataset = get_train_data(california_dataset)
valid_ftmyers_dataset, train_ftmyers_dataset = get_train_data(ftmyers_dataset)
valid_hyspiri_17_dataset, train_hyspiri_17_dataset = get_train_data(hyspiri_17_dataset)
valid_hyspiri_40_dataset, hyspiri_40_dataset = get_train_data(hyspiri_40_dataset)
valid_amikougami_dataset, train_amikougami_dataset = get_train_data(amikougami_dataset)
valid_arizona_dataset, train_arizona_dataset = get_train_data(arizona_dataset)
valid_campbellriver_dataset, train_campbellriver_dataset = get_train_data(campbellriver_dataset)
valid_dewar_dataset, train_dewar_dataset = get_train_data(dewar_dataset)
valid_lethbridge_dataset, train_lethbridge_dataset = get_train_data(lethbridge_dataset)
valid_qccut_dataset, train_qccut_dataset = get_train_data(qccut_dataset)
valid_seney_dataset, train_seney_dataset = get_train_data(seney_dataset)
valid_yellowstone_dataset, train_yellowstone_dataset = get_train_data(yellowstone_dataset)
valid_dataset_list = (valid_california_dataset + valid_ftmyers_dataset + valid_hyspiri_17_dataset
                      + valid_hyspiri_40_dataset + valid_amikougami_dataset + valid_arizona_dataset
                      + valid_campbellriver_dataset + valid_dewar_dataset + valid_lethbridge_dataset
                      + valid_qccut_dataset + valid_seney_dataset + valid_yellowstone_dataset)
train_dataset_list = [item for item in file_list if item not in valid_dataset_list]
for item in valid_dataset_list:
    if item in train_dataset_list:
        print(True)
print('Moffett10.mat' in train_dataset_list)
valid_5HSI_dataset = [random.choice(valid_california_dataset)]
valid_5HSI_dataset.append(random.choice(valid_ftmyers_dataset))
valid_5HSI_dataset.append(random.choice(valid_hyspiri_17_dataset))
valid_5HSI_dataset.append(random.choice(valid_hyspiri_40_dataset))
for item in valid_5HSI_dataset:
    print(item in train_dataset_list)
valid_5HSI_dataset.append('Moffett10.mat')
with open('../testpath/5HSI_test.txt', mode='w') as file:
    file.write('\n'.join(valid_5HSI_dataset))
print(len(valid_dataset_list))
print(len(train_dataset_list))
with open('../testpath/5HSI_test.txt', mode='r') as file:
    print(file.readlines())
try:
    with open('../trainpath/total_train.txt', mode='w') as file:
        file.write('\n'.join(train_dataset_list))
    with open('../testpath/total_valid.txt', mode='w') as file:
        file.write('\n'.join(valid_dataset_list))
except Exception as e:
    print('During handling file process, an exception came out.')

