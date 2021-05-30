# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:07:02 2021

@author: Lenovo

输入：
将51*22的“模型predict”出来的向量（center_vali_unprocessed.npy）放在当前目录

输出：
在当前目录生成51*11*2的映射到原图尺寸的center_vali_processed.npy


"""

import torch
import numpy as np
import glob,random
import cv2
from PIL import Image

#读取图片，从而知道映射需要用到的图片的尺寸
picpath = r'./'

#从224映射到原图
def coord_mapping(picpath):
    dirt = glob.glob(picpath+'/test/data/*.jpg')
    
    center = np.load('center_vali_unprocessed.npy')
    center = center.reshape(-1,11,2)
    
    for i,e in enumerate(dirt):
        img = np.fromfile(e, np.uint8)
        img = cv2.imdecode(img, 1)#图片大小为(高，宽,通道)(在工作区中看到的np矩阵)
        
        center[i,:,0]*=img.shape[1]/224
        center[i,:,1]*=img.shape[0]/224
    
    center = np.array(center , dtype = int)
    
    np.save('center_vali_processed.npy',center)
        


coord_mapping(picpath)
    