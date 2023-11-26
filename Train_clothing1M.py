from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=8, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.0009, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='./Dataset/noise_label_data/Clothing_Ori', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=15625, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import cv2
import math
def attention_erase_map(images, outputs, gmmweight):
    erase_x=[]
    erase_y=[]
    erase_x_min=[]
    erase_y_min=[]
    width=images.shape[2]
    height=images.shape[3]
    outputs = (outputs**2).sum(1)
    b, h, w = outputs.size()#shape
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)
    for j in range(outputs.size(0)):
        am = outputs[j, ...].detach().cpu().numpy()
        am = cv2.resize(am, (width, height))
        am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        m=np.argmax(am)
        m_min=np.argmin(am)
        r, c = divmod(m, am.shape[1])
        rmin, cmin = divmod(m_min, am.shape[1])
        erase_x.append(r)
        erase_y.append(c)
        erase_x_min.append(rmin)
        erase_y_min.append(cmin)

    erase_x=torch.tensor(erase_x).cuda()
    erase_y=torch.tensor(erase_y).cuda()
    erase_x_min=torch.tensor(erase_x_min).cuda()
    erase_y_min=torch.tensor(erase_y_min).cuda()
    sl = 0.06
    sh = 0.13
    r1 = 0.8
    img=images.clone()
    img_min=images.clone()
    erase_x = []
    erase_u = []
    for i in range(img.size(0)):
        for attempt in range(1000000000):
            area = img.size()[2] * img.size()[3]
            target_area = random.uniform(sl, sh) *area + 0.1 * (1-gmmweight[i])
            aspect_ratio = random.uniform(r1, 1 / r1)

            target_area_min = random.uniform(sl, sh) *area + 0.1 * (1-gmmweight[i])
            aspect_ratio_min = random.uniform(r1, 1 / r1)



            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            hmin = int(round(math.sqrt(target_area_min * aspect_ratio_min)))
            wmin = int(round(math.sqrt(target_area_min / aspect_ratio_min)))

            if w < img.size()[3] and h < img.size()[2] and wmin < img.size()[3] and hmin < img.size()[2]:
                erase_x.append(target_area)
                erase_u.append(target_area_min)
                x1 = erase_x[i]
                y1 = erase_y[i]
                x1_min = erase_x_min[i]
                y1_min = erase_y_min[i]
                if x1+h>img.size()[2]:
                    x1=img.size()[2]-h
                if y1+w>img.size()[3]:
                    y1=img.size()[3]-w
                if x1_min+hmin>img.size()[2]:
                    x1_min=img.size()[2]-hmin
                if y1_min+wmin>img.size()[3]:
                    y1_min=img.size()[3]-wmin

                if img.size()[1] == 3:
                    img[i, 0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)

                    img_min[i, 0, x1_min:x1_min + hmin, y1_min:y1_min + wmin] = random.uniform(0, 1)
                    img_min[i, 1, x1_min:x1_min + hmin, y1_min:y1_min + wmin] = random.uniform(0, 1)
                    img_min[i, 2, x1_min:x1_min + hmin, y1_min:y1_min + wmin] = random.uniform(0, 1)
                    break
    return erase_x, erase_u, img, img_min





# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, w_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, w_u = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        w_u = w_u.view(-1,1).type(torch.FloatTensor)
        inputs_x, inputs_x2, labels_x, w_x, w_u = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), w_u.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            maps_u, outputs_u11 = net(inputs_u, return_map=True)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            maps_x, outputs_x = net(inputs_x, return_map=True)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       


        with torch.no_grad():
            inputs_erase = torch.cat([inputs_x, inputs_u], dim=0)
            inputs_map = torch.cat([maps_x, maps_u], dim=0)
            gmm_weight = torch.cat([w_x, w_u], dim=0)
            erase_ratio_max, erase_ratio_min, erase_imgx_max, erase_imgx_min = attention_erase_map(inputs_erase, inputs_map, gmm_weight)
            hot_label_x = torch.argmax(targets_x, dim=1)
            hot_label_u = torch.argmax(targets_u, dim=1)
            erase_label_x_max = torch.zeros(targets_x.shape)
            erase_label_x_min = torch.zeros(targets_x.shape)
            erase_label_u_max = torch.zeros(targets_u.shape)
            erase_label_u_min = torch.zeros(targets_u.shape)
            for p in range(hot_label_x.shape[0]):
                erase_label_x_max[p] = torch.Tensor([erase_ratio_max[:batch_size][p]/13]).repeat(14)
                erase_label_x_min[p] = torch.Tensor([erase_ratio_min[:batch_size][p]/13]).repeat(14)
                erase_label_x_max[p][hot_label_x[p]] = -erase_ratio_max[:batch_size][p]
                erase_label_x_min[p][hot_label_x[p]] = -erase_ratio_min[:batch_size][p]
            for p in range(hot_label_u.shape[0]):
                erase_label_u_max[p] = torch.Tensor([erase_ratio_max[batch_size:][p]/13]).repeat(14)
                erase_label_u_min[p] = torch.Tensor([erase_ratio_min[batch_size:][p]/13]).repeat(14)
                erase_label_u_max[p][hot_label_u[p]] = -erase_ratio_max[batch_size:][p]
                erase_label_u_min[p][hot_label_u[p]] = -erase_ratio_min[batch_size:][p]
            targets_max_x = targets_x + erase_label_x_max.cuda()
            targets_min_x = targets_x + erase_label_x_min.cuda()
            targets_max_u = targets_u + erase_label_u_max.cuda()
            targets_min_u = targets_u + erase_label_u_min.cuda()


        reconimg = torch.cat([erase_imgx_max, erase_imgx_min], dim=0)
        reconout = net(reconimg, recon=True)
        Lrecon = F.mse_loss(reconout[:inputs_erase.shape[0]], inputs_erase, size_average=True) + F.mse_loss(reconout[inputs_erase.shape[0]:], inputs_erase, size_average=True)

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)        
        all_inputs = torch.cat([inputs_x, inputs_x2, erase_imgx_max[:batch_size], erase_imgx_min[:batch_size], inputs_u, inputs_u2, erase_imgx_max[batch_size:], erase_imgx_min[batch_size:]], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_max_x, targets_min_x, targets_u, targets_u, targets_max_u, targets_min_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*4] + (1 - l) * input_b[:batch_size*4]        
        mixed_target = l * target_a[:batch_size*4] + (1 - l) * target_b[:batch_size*4]
        logits = net(mixed_input)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty + Lrecon * 0.3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Acc: %.2f%%" %(k,acc))  
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
    return acc

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)       
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    
    
def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths  
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
from resnetc2d import *      
def create_model(net='resnet50', num_class=14):
    chekpoint = torch.load('./pretrained/ckpt_clothing_{}.pth'.format(net))
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet(net, num_classes=num_class, pool=True)
    model.load_state_dict(sd, strict=False)
    model = model.cuda()
    return model   

log=open('./check_clothing1m'+'/checkpoint/label1%s.txt'%args.id,'w')     
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

net3 = create_model()
net4 = create_model()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<1:     # warm up  
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)              # train net2
    
    val_loader = loader.run('val') # validation
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1 = eval_train(epoch,net1) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2 = eval_train(epoch,net2) 
    test_loader = loader.run('test')
    acc = test(net1,net2,test_loader)
    print("test accuracy:", acc)
test_loader = loader.run('test')
net1.load_state_dict(torch.load('./check_clothing1m'+'/checkpoint/label1%s_net1.pth.tar'%args.id))
net2.load_state_dict(torch.load('./check_clothing1m'+'/checkpoint/label1%s_net2.pth.tar'%args.id))
acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush() 
