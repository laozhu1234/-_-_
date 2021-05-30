# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:05:32 2021

@author: Lenovo_ztk
"""

import os,shutil

root = os.getcwd()
#为.\VGG2分类与5分类

classes = [str(i) for i in range(0,7)]
oldlabels = ['train', 'test', 'predicted test']
newlabel5 = ['new train', 'test', 'predicted test_5']
newlabel2 = ['new train', 'test', 'predicted test_2']

for j in range(3):
    foldername = os.path.join(root,'五分类',newlabel5[j])
    if not os.path.exists(foldername):
            os.makedirs(foldername)    
            
    foldername = os.path.join(root,'二分类',newlabel2[j])
    if not os.path.exists(foldername):
            os.makedirs(foldername) 
            
    for i in classes:     
        src = os.path.join(root,oldlabels[j],i)
        
        if(int(i)<5):
            dst = os.path.join(root,'五分类',newlabel5[j],i)
        else:
            dst = os.path.join(root,'二分类',newlabel2[j],i)
        
        os.rename(src, dst)
