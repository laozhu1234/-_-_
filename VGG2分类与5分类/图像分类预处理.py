import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float
import numpy as np
import os
import glob
import cv2
import math
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#以下函数中的label参数的输入为'train' 或 'test'
#创建类别标签文件夹
def mkdir(label):
    print('start making')
    for i in range(7):
        la = str(i)
        path = './' + label + '/' + la
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
    print('finish making')

#获取原始坐标并存入numpy
def load_center(classes,path,label):
    print('start loading ' + label + 'ing data')
    text = glob.glob(path+'/'+label+'/data/*.txt')
    num = len(text)
    cen = np.zeros((num, 11, 2))
    for index in range(num):
        for line in open(text[index]):
            pos=classes[(line.split("'identification': '")[1].split("'")[0])]
            cen[index][pos][0] = int(line.split(",")[0])
            cen[index][pos][1] = int(line.split(",")[1])
    np.save('center_' + label + '_origin.npy', cen) #准确的坐标（其实这个文件回归任务时已经生成过了)
    print('finish loading')

#获取分割图像并存入numpy
def get_parts(path,label):
    print('start trimming ' + label + 'ing images')
    pic = glob.glob(path + '/' + label + '/data/*.jpg')
    center = np.load('center_' + label + '_origin.npy')
    center = np.array(center, dtype=int)
    num = len(pic)
    img_parts = np.zeros((num,11,32,32,3))
    for index in range(num):
        img_org1 = cv2.imread(pic[index], 1)#读取图片
        img_gamma1 = exposure.adjust_gamma(img_org1, 0.4)#进行伽马校正
        #计算对应框的宽和高
        for i in range(11):
            if i==0:
                dist = center[index][0] - center[index][1]
                h = math.hypot(dist[0], dist[1])*2
                w = 1.4 * h
            elif i==10:
                dist = center[index][9] - center[index][10]
                h = math.hypot(dist[0], dist[1]) * 2
                w = 1.4 * h
            else:
                dist = center[index][i-1] - center[index][i+1]
                h = math.hypot(dist[0], dist[1])
                w = 1.4 * h
            img_gamma2 = img_gamma1[center[index][i][1] - int(h / 2):center[index][i][1] + int(h / 2),
                         center[index][i][0] - int(w / 2):center[index][i][0] + int(w / 2),:]#获取切割出的图像
            img_gamma3 = cv2.resize(img_gamma2, dsize=(32, 32))#resize成32*32
            img_parts[index][i] = img_gamma3#存入numpy
    np.save('img_parts_' + label + '.npy', img_parts)   #根据坐标割取出来的图片,(num*32*32*3)
    print('finish trimming')

#获取每张图像的11个类别标签并存入numpy
def get_label(classes,path,label):
    print('start getting ' + label + 'ing labels')
    text = glob.glob(path + '/' + label + '/data/*.txt')
    num = len(text)
    la = np.zeros((num,11))
    #五分类的五个类用0、1、2、3、4表示，二分类的两个类用5、6表示
    for index in range(num):
        for line in open(text[index]):
            pos = classes[(line.split("'identification': '")[1].split("'")[0])]
            if pos%2 == 0:
                la[index][pos] = int(line.split("'disc': 'v")[1][0]) - 1
            else:
                la[index][pos] = int(line.split("'vertebra': 'v")[1][0]) + 4
    np.save('label_' + label + '.npy', la)  #(num*11),每张图片的各部分所属类别
    print('finish')

#将分割出来的图像放入对应类别标签的文件夹
def sort_parts(label):
    print('start sorting ' + label + 'ing imgs')
    parts = np.load('img_parts_' + label + '.npy')
    labels = np.load('label_' + label + '.npy')
    c = 0
    for i in range(parts.shape[0]):
        for j in range(parts.shape[1]):
            img = Image.fromarray(np.uint8(parts[i][j]))#将numpy转为图像
            img.save('./' + label + '/' + str(int(labels[i][j])) + '/' + str(c) + '.jpg')#保存在对应类别文件夹中
            c += 1
    print('finish sorting ' + label + 'ing imgs')

if __name__ == '__main__':
    classes = {'T12-L1': 0, 'L1': 1, 'L1-L2': 2, 'L2': 3, 'L2-L3': 4, 'L3': 5, 'L3-L4': 6, 'L4': 7, 'L4-L5': 8, 'L5': 9,
               'L5-S1': 10}
    # path为文件路径
    # 运行时只需将path改为数据所在的路径
    path = './'
    #生成的文件夹或目录均在该py文件所在的文件夹
    mkdir('train')
    mkdir('test')
    load_center(classes,path,'train')
    load_center(classes,path,'test')
    get_parts(path, 'train')
    get_parts(path, 'test')
    get_label(classes, path, 'train')
    get_label(classes, path, 'test')
    sort_parts('train')
    sort_parts('test')


