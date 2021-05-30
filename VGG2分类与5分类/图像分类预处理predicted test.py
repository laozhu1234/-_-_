'''
2021/5/26
'''

import torch
from skimage import exposure
import random, os, glob
import cv2,math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def mkdir(label):
    print('start making')
    for i in range(7):
        la = str(i)
        path = './' + label + '/' + la
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
    print('finish making')
    
#图片->割取->npy
def get_parts(path): 
    print('start trimming ' + 'predicted test' + 'ing images')
    
    pic = glob.glob(path + '/' + 'test' + '/data/*.jpg')
    center = np.load('center_' + 'vali' + '_processed.npy') #51*11*2
    center = np.array(center, dtype=int)
    
    num = len(pic)
    img_parts = np.zeros((num,11,32,32,3))
    
    # savepath = r'C:\Users\Lenovo\.spyder-py3\脊椎\回归割图可视化\割得图\temp'
    
    for index in range(num):
        img_org1 = np.fromfile(pic[index], np.uint8)
        img_org1 = cv2.imdecode(img_org1, 1) #图片大小为(高，宽,通道)(在工作区中看到的np矩阵)
        # cv2.imshow('org pic', img_org1)
        
        img_gamma1 = exposure.adjust_gamma(img_org1, 0.4)
        # cv2.imshow('gamma', img_gamma1)
        
        name = pic[index].split('\\')[-1]
        name = name[0:-4]
        #割图
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
            
            u = max(0, center[index][i][1] - int(h / 2))
            d = min(img_gamma1.shape[0], center[index][i][1] + int(h / 2))
            l = max(0, center[index][i][0] - int(w / 2))
            r = min(img_gamma1.shape[1], center[index][i][0] + int(w / 2))
            img_gamma2 = img_gamma1[u:d, l:r ,:]         
            # cv2.imshow('org crop', img_gamma2)                
            
            img_gamma3 = cv2.resize(img_gamma2, dsize=(32, 32))
            img_parts[index][i] = img_gamma3
            # cv2.imshow('resized crop', img_gamma3)
            
            # temp_img = Image.fromarray(np.uint8(img_gamma3))
            # temp_img.save(savepath + '\\' + name + ' _' + str(i) + '.jpg')
            
    np.save('img_parts_' + 'predicted test' + '.npy', img_parts)
    print('finish trimming')


def get_name(path):
    dirt = glob.glob(path + '/' + 'test' + '/data/*.jpg')
    name = []
    
    for i, e in enumerate(dirt):
        name.append(e.split('\\')[-1])
        name[i] = name[i][0:-4]
    name = np.array(name)
    return name

def sort_parts():
    
    print('start sorting ' + 'predicted test' + 'ing imgs')
    
    parts = np.load('img_parts_' + 'predicted test' + '.npy')
    labels = np.load('label_' + 'test' + '.npy')
    name = get_name(path)
    
    c = 0
    for i in range(parts.shape[0]): #图片编号，共147张（训练集）
        for j in range(parts.shape[1]): #11个位置
            img = Image.fromarray(np.uint8(parts[i][j]))
            savepath = './' + 'predicted test' + '/' + str(int(labels[i][j]))+ '/' + name[i] + '_' + str(j)  + '.jpg'
            img.save(savepath)
            c += 1
            
    print('finish sorting ' + 'predicted test' + 'ing imgs')


if __name__ == '__main__':
    classes = {'T12-L1': 0, 'L1': 1, 'L1-L2': 2, 'L2': 3, 'L2-L3': 4, 'L3': 5, 'L3-L4': 6, 'L4': 7, 'L4-L5': 8, 'L5': 9,
               'L5-S1': 10}
    
    #源图片路径
    path = r'./'   
    
    mkdir('predicted test')
    
    get_parts(path)
    
    parts_test = np.load('img_parts_test.npy')
    label_test = np.load('label_test.npy')
    
    sort_parts()
    
    print(parts_test.shape)
    print(label_test.shape)
    parts_test = np.array(parts_test,dtype=int)
    plt.imshow(parts_test[0][0])
    plt.show()
