# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:26:11 2021

@author: Lenovo_ztk
"""

import os,shutil

root = os.path.abspath(os.path.join(os.getcwd(),'..'))

origin_path = root + r'\目标检测\center_test_predicted.npy'
new_file_name = '.\center_vali_unprocessed.npy'
shutil.copyfile(origin_path,new_file_name)  #复制文件