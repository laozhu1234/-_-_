import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import io, time
from PIL import Image,ImageDraw
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#请在此处设定参数
#img_path1和img_path2中'\test'之后的内容一般不需要改动！
img_path1 = r'.\test\data\*.jpg' #用于抓取文件名
img_path2 = r'.\test\show\show_' #用于读取标记图
draw_path = 'test_visable/' #注意要在当前目录新建一个文件夹
cood_path = 'center_test_predicted.npy'
dic_path = 'test_dic.plk'


def load(path):
    with open(path,'rb') as f:
       data = pickle.load(f)
    return data

def Label(filename,coord):   
    '''
    一次处理一张图片
    '''   
    #对字符串变换
    s1=img_path2
    s2=filename #e.g. study10.jpg
    filename = s1+s2
    
    imgid=s2[5:-4]  #e.g. study10
    
    coord=np.reshape(coord, [11,2]) #一行一个坐标
    with Image.open(filename) as img: 
        coord[:,0]=coord[:,0]/224*img.size[0]   #img的第0维为宽
        coord[:,1]=coord[:,1]/224*img.size[1]
        coord=np.array(coord,dtype=int)
        
        draw = ImageDraw.Draw(img)
        for i,pair in enumerate(coord):      
            x,y = pair[0],pair[1]
            scale=5             #可以理解为半径

            draw.arc(((x-scale,y-scale),(x+scale,y+scale)), 0,360, fill=255)    #画空心圆
            
            draw.text((x,y),str(i+1),fill=255)  #坐标，字符，颜色
            
        img.save(draw_path+'my'+imgid+'.jpg')
        
if __name__ == '__main__':
    print('开始建立字典')
    
    centers_test = np.load(cood_path)
    
    pic_path=glob.glob(img_path1)
    dic={}
    for i in range(len(pic_path)):
        t_label=pic_path[i].split('data\\')[-1]
        dic[t_label]=centers_test[i]
     
 
    pickle.dump(dic,open(dic_path,'wb'))
    print('完成建立，开始标记')

    data=load(dic_path)
    
    for item in data:
        Label(item, data[item])








