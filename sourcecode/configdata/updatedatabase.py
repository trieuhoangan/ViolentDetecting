import sys
import os

folder_path = 'newspaper'
file_path = 'newspapers/1712.txt.pro'
with open(file_path,'r',encoding = 'utf-8',errors ='replace') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        print('-----------------------------------------------')