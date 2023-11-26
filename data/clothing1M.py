from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pandas as pd
import cv2


class clothing1M_dataset(data.Dataset):
    def __init__(self, root, label_file='noisy_train.txt', transform=None, mode='None'):
        self.root = os.path.expanduser(root)
        self.transform = transform

        ##'T-Shirt','Shirt','Knitwear','Chiffon','Sweater','Hoodie','Windbreaker','Jacket','Downcoat','Suit','Shawl','Dress','Vest','Underwear'
        file = open(label_file, 'r')
        self.gt_all = file.readlines()
        self.img_name = [l.split()[0] for l in self.gt_all]#[:200000]
        self.label = [l.split()[1] for l in self.gt_all]#[:200000]
        self.mode=mode
    def __getitem__(self, index):
        #img = cv2.imread(self.root + self.img_name[index])
        img = Image.open(self.root + self.img_name[index]).convert('RGB')
        if img is None:
            img = cv2.imread(self.root + self.img_name[index+1])
        target = int(self.label[index])
        if self.mode=='labeled':
           img1 = self.transform(img)
           img2 = self.transform(img)  
           return img1,img2, target,0.8
        if self.mode=='unlabeled':
           img1 = self.transform(img)
           img2 = self.transform(img)
           return img1,img2
        if self.mode=='all':
           img1 = self.transform(img)
           return img1,target,index
        if self.mode=='test' or self.mode=='val':
           img1 = self.transform(img)
           return img1,target


    def __len__(self):
        return len(self.label)

