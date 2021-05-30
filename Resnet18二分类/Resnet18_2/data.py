import pickle
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import random

random.seed(0)

filenum = 0
pix = np.zeros([162 + 574, 32, 32, 3])
path = r"./data_part2/train"
for num in range(5, 7):
    dir = os.path.join(path, str(num))
    files = os.listdir(dir)
    Total_num = len(files)
    for file in files:
        file_dir = os.path.join(dir, file)
        # print("loading file name：", file_dir)
        img = Image.open(file_dir)  # 读取图片
        img = np.array(img)
        # plt.imshow(img)
        # plt.show()
        pix[filenum] = img
        # print(pix[filenum].shape)
        # plt.imshow(pix[filenum] / 255)
        # plt.show()
        filenum += 1

    print("Done file:" + str(num + 1))

label_0 = np.zeros(162)
label_1 = np.ones(574)
label = np.append(label_0, label_1)
pix = pix.transpose(0, 3, 1, 2)

a = np.arange(0, 162+574, dtype=int)
random.shuffle(a)
pix[:] = pix[a]
label[:] = label[a]

np.save('train_label.npy', label)
np.save('train_img.npy', pix)

# test
filenum = 0

pix = np.zeros([26 + 229, 32, 32, 3])
path = r"./data_part2/test"
for num in range(5, 7):
    dir = os.path.join(path, str(num))
    files = os.listdir(dir)
    Total_num = len(files)
    for file in files:
        file_dir = os.path.join(dir, file)
        # print("loading file name：", file_dir)
        img = Image.open(file_dir)  # 读取图片
        img = np.array(img)
        # plt.imshow(img)
        # plt.show()
        pix[filenum] = img
        # print(pix[filenum].shape)
        # plt.imshow(pix[filenum] / 255)
        # plt.show()
        filenum += 1

    print("Done file:" + str(num + 1))
label_0 = np.zeros(26)
label_1 = np.ones(229)
label = np.append(label_0, label_1)
pix = pix.transpose(0, 3, 1, 2)

a = np.arange(0, 26 + 229, dtype=int)
random.shuffle(a)
pix[:] = pix[a]
label[:] = label[a]

print(pix.shape)
print(label.shape)

np.save('test_label.npy', label)
np.save('test_img.npy', pix)
