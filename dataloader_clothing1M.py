from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
from autoaugment import CIFAR10Policy, ImageNetPolicy
transform_weak_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)


def transform_weak_c1m(x):
    return transform_weak_c1m_c10_compose(x)


transform_strong_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)


def transform_strong_c1m_c10(x):
    return transform_strong_c1m_c10_compose(x)


transform_strong_c1m_in_compose = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ]
)
def transform_strong_c1m_in(x):
    return transform_strong_c1m_in_compose(x)
class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}      
        self.root='./Dataset/noise_label_data/Clothing1M/'
        file_train = open("./Dataset/noise_label_data/Clothing1M/noisy_train.txt", 'r')
        self.gt_all_train = file_train.readlines()
        self.img_name_train = [l.split()[0] for l in self.gt_all_train]
        self.label_train = [int(l.split()[1]) for l in self.gt_all_train]
        file_test = open("./Dataset/noise_label_data/Clothing1M/clean_test.txt", 'r')
        self.gt_all_test = file_test.readlines()
        self.img_name_test = [l.split()[0] for l in self.gt_all_test]
        self.label_test = [int(l.split()[1]) for l in self.gt_all_test]
        

        file_val = open("./Dataset/noise_label_data/Clothing1M/clean_val.txt", 'r')
        self.gt_all_val = file_val.readlines()
        self.img_name_val = [l.split()[0] for l in self.gt_all_val]
        self.label_val = [int(l.split()[1]) for l in self.gt_all_val]
        print(len(self.label_train))



        for i in range(len(self.img_name_train)):
            self.train_labels[self.img_name_train[i]]=self.label_train[i]
        for i in range(len(self.img_name_test)):
            self.test_labels[self.img_name_test[i]]=self.label_test[i]
        for i in range(len(self.img_name_val)):
            self.val_labels[self.img_name_val[i]]=self.label_val[i]

        if mode == 'all':
            train_imgs=self.img_name_train                             
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)       
        elif self.mode == "labeled":   
            train_imgs = self.img_name_train
            pred_idx = pred.nonzero()[0]
            print(pred_idx.shape)
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = self.img_name_train
            pred_idx = (1-pred).nonzero()[0]  
            print(pred_idx.shape)
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='test':
            self.test_imgs=self.img_name_test
        elif mode=='val':
            self.val_imgs =self.img_name_val
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob  
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, prob
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
class clothing_dataloader():  
    def __init__(self, root, batch_size, num_batches, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])        
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='test':
            test_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             
        elif mode=='val':
            val_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader     
