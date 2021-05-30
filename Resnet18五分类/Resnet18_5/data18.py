import pickle
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

random.seed(0)

filenum = 0

class_0 = 407
class_1 = 221
class_2 = 218
class_3 = 31
class_4 = 5
all_class = class_0 + class_1 + class_2 + class_3 + class_4
# all_class = class_0 + class_1 + class_2
pix = np.zeros([all_class, 32, 32, 3])

path = r"./data_part5/train"
for num in range(5):
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
label_0 = np.zeros(class_0)
label_1 = np.ones(class_1)
label_2 = np.ones(class_2) * 2
label_3 = np.ones(class_3) * 3
label_4 = np.ones(class_4) * 4
label = np.append(label_0, label_1)
label = np.append(label, label_2)
label = np.append(label, label_3)
label = np.append(label, label_4)
pix = pix.transpose(0, 3, 1, 2)

a = np.arange(0, all_class, dtype=int)
random.shuffle(a)
pix[:] = pix[a]
label[:] = label[a]

print(pix.shape)
print(label.shape)

np.save('train_label.npy', label)
np.save('train_img.npy', pix)

# test
filenum = 0
pix = np.zeros([306, 32, 32, 3])
path = r"./data_part5/test"
for num in range(5):
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
label_0 = np.zeros(118)
label_1 = np.ones(74)
label_2 = np.ones(62) * 2
label_3 = np.ones(16) * 3
label_4 = np.ones(36) * 4

label = np.append(label_0, label_1)
label = np.append(label, label_2)
label = np.append(label, label_3)
label = np.append(label, label_4)
pix = pix.transpose(0, 3, 1, 2)

a = np.arange(0, 306, dtype=int)
random.shuffle(a)
pix[:] = pix[a]
label[:] = label[a]

print(pix.shape)
print(label.shape)

np.save('test_label.npy', label)
np.save('test_img.npy', pix)
