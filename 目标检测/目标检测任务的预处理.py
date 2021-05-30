import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data, exposure, img_as_float

#label为'train' 或 'test'
def load_txt(classes,path,label):
    print('start loading ' + label + 'ing data')
    text=glob.glob(path+'/'+label+'/data/*.txt')
    pic=glob.glob(path+'/'+label+'/data/*.jpg')
    num = len(pic)#图片数量
    img = np.zeros((num, 224, 224, 3))#保存图像的numpy
    cen_org = np.zeros((num, 11, 2))  # 保存新坐标的numpy
    cen = np.zeros((num, 11, 2))#保存新坐标的numpy
    scale = np.zeros((num,2))#保存横纵比例的numpy
    for index in range(num):
        x = cv2.imread(pic[index], 1)
        hp, wp = 224 / x.shape[0], 224 / x.shape[1]#计算横纵比例
        x = cv2.resize(x,dsize = (224, 224))#resize为224*224
        x = exposure.adjust_gamma(x, 0.4)#伽马校正
        mean = np.mean(x, axis=(0, 1)).T#计算每个通道平均值
        var = np.std(x, axis=(0, 1)).T#计算每个通道方差
        x = (x - mean) / var#得到新的图片
        img[index] = x
        scale[index][0] = wp
        scale[index][1] = hp
        # 遍历该txt文件的每一行
        for line in open(text[index]):
            identification = (line.split("'identification': '")[1].split("'")[0])  # 提取该行的identification
            pos = classes[identification]  # 得到此identification在cen中对应的位置
            cen[index][pos][0] = int(line.split(",")[0]) * wp  # 得到新的横坐标
            cen[index][pos][1] = int(line.split(",")[1]) * hp  # 得到新的纵坐标
            cen_org[index][pos][0] = int(line.split(",")[0])  # 得到原图的横坐标
            cen_org[index][pos][1] = int(line.split(",")[1])  # 得到原图的纵坐标
    np.save('img_' + label + '.npy', img)       #224*224*3的图像npy文件
    np.save('center_' + label + '.npy', cen)    #224*224尺度下的中心坐标
    np.save('scale_' + label + '.npy', scale)   #比例，224除以原图尺寸
    np.save('center_org_' + label + '.npy', cen_org)    #原图尺寸下的中心坐标
    print('finish loading')

if __name__ == '__main__':
    classes={'T12-L1':0,'L1':1,'L1-L2':2,'L2':3,'L2-L3':4,'L3':5,'L3-L4':6,'L4':7,'L4-L5':8,'L5':9,'L5-S1':10}#从上到下确定位置
    # path为文件路径
    #运行时只需将path改为作业数据所在的路径
    path='./'
    # 生成的文件夹或目录均在该py文件所在的文件夹
    load_txt(classes, path, 'train')
    load_txt(classes, path, 'test')
    img_train = np.load('img_train.npy')
    img_test = np.load('img_test.npy')
    center_train = np.load('center_train.npy')
    center_test = np.load('center_test.npy')
    plt.imshow(img_train[0])#显示一张处理后的图片
    plt.show()
