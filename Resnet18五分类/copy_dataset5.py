# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:05:32 2021

@author: Lenovo_ztk
"""

import os,shutil

root = os.path.abspath(os.path.join(os.getcwd(),'..'))
#为.\根目录

classes = [str(i) for i in range(0,5)]
oldlabel5 = ['new train', 'predicted test_5']
newlabel5 = ['train', 'test']

for j in range(2):
    foldername = os.path.join(root,'Resnet18五分类\Resnet18_5\data_part5',newlabel5[j])
    if not os.path.exists(foldername):
            os.makedirs(foldername)           
            
    for i in classes:     
        src = os.path.join(root,'VGG2分类与5分类\五分类',oldlabel5[j],i)
        
        dst = os.path.join(root,'Resnet18五分类\Resnet18_5\data_part5',newlabel5[j],i)
        
        shutil.copytree(src, dst)
