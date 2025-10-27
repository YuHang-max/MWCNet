import os
import random
from scipy.io import loadmat

# data = ['HSIv2/California/data_14.mat\n', 'HSIv2/California/data_24.mat\n', 'HSIv2/California/data_53.mat\n',
#         'HSIv2/California/data_102.mat\n', 'HSIv2/California/data_30.mat\n', 'HSIv2/California/data_12.mat\n',
#         'HSIv2/California/data_18.mat\n', 'HSIv2/California/data_8.mat\n']
#
# f = open('testpath/val_fig.txt', 'a')
# for mat in data:
#     f.write(mat)

prefix = 'dataset/AVIRIS/test/hsi'
results = os.walk('..')
names = []
for root, dirs, files in os.walk(prefix, topdown=False):
    for file in files:
        if '.mat' in file:
            names.append(file)

print(len(names), names)
num = len(names)
# testname = random.sample(names, round(0.1 * num))
# print(len(testname), testname)
# for name in testname:
#     names.remove(name)


# for trainname in names:
#     f = open('trainpath/train.txt', 'a')
#     if '.mat' in trainname:
#         f.write(os.path.join(prefix, trainname).replace('\\', '/'))
#     f.write('\n')

for test in names:
    f = open('../testpath/test.txt', 'a')
    if '.mat' in test:
        f.write(os.path.join(prefix, test).replace('\\', '/'))
    f.write('\n')

# data = 'data_17.mat'
# data = loadmat(data)
# print(data['Xim'].shape)



