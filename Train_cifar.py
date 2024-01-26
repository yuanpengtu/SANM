from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from resnetc2d import *
import random
import cv2
import math
def attention_erase_map(images, outputs, gmmweight, target_mask):
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
    sl = 0.02
    sh = 0.07
    r1 = 0.8
    img=images.clone()
    img_min=images.clone()
    for i in range(img.size(0)):
        for attempt in range(1000000000):
            area = img.size()[2] * img.size()[3]
            target_area_ratio =  random.uniform(sl, sh) + 0.03 * (1-gmmweight[i])
            target_area = target_area_ratio*area
            maxindex = torch.argmax(target_mask[i])
            flagmask = torch.zeros(target_mask[i].shape)
            flagmask[maxindex] = 1
            flagmask = flagmask.bool()
            target_mask[i][flagmask] -= target_area_ratio
            target_mask[i][~flagmask] += target_area_ratio/(target_mask[i].shape[-1]-1)
            aspect_ratio = random.uniform(r1, 1 / r1)
            h = int(round(math.sqrt(target_area*0.5 * aspect_ratio)))
            w = int(round(math.sqrt(target_area*0.5 / aspect_ratio)))
            if w < img.size()[3] and h < img.size()[2]:
                x1 = erase_x[i]
                y1 = erase_y[i]
                x1_min = erase_x_min[i]
                y1_min = erase_y_min[i]
                if x1+h>img.size()[2]:
                    x1=img.size()[2]-h
                if y1+w>img.size()[3]:
                    y1=img.size()[3]-w
                if x1_min+h>img.size()[2]:
                    x1_min=img.size()[2]-h
                if y1_min+w>img.size()[3]:
                    y1_min=img.size()[3]-w

                if img.size()[1] == 3:
                    img[i, 0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)

                    img_min[i, 0, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    img_min[i, 1, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    img_min[i, 2, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    break
    return img, img_min, target_mask


def js_loss_compute(pred_prob, target, num_classes, reduce=True):
    ## pred_prob: [B,C]; target(int): [B,]
    ones = torch.sparse.torch.eye(num_classes).cuda()
    print(target.long())
    label_one_hot = ones.index_select(0,target.long())

    kl_1 = F.kl_div(torch.log(pred_prob + 1e-14), (pred_prob + label_one_hot)/2., reduce=False)
    kl_2 = F.kl_div(torch.log(label_one_hot + 1e-14), (pred_prob + label_one_hot)/2., reduce=False)
    js = (kl_1 + kl_2)/2.
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)
def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=360, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--check_path', default='./check_cifar-10', type=str, help='path to checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
num_classes=args.num_class

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval()
    conf_penalty = NegEntropy()
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_xp, inputs_xp2, inputs_xs1, inputs_xs2, inputs_x, inputs_x2, labels_x, truelab, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_up, inputs_up2, inputs_us1, inputs_us2,inputs_u,inputs_u2,w_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_up, inputs_up2, inputs_us1, inputs_us2,inputs_u,inputs_u2,w_u = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        labels_xnew=labels_x
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        w_u = w_u.view(-1,1).type(torch.FloatTensor) 
        inputs_x, inputs_x2, labels_x, w_x, w_u = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), w_u.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        inputs_us1,inputs_xs1 = inputs_us1.cuda(),inputs_xs1.cuda()
        inputs_us2,inputs_xs2 = inputs_us2.cuda(),inputs_xs2.cuda()
        inputs_up,inputs_xp = inputs_up.cuda(),inputs_xp.cuda()
        inputs_up2,inputs_xp2 = inputs_up2.cuda(),inputs_xp2.cuda()
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            map_u, _, outputs_u11 = net(inputs_u,return_map=True)
            _, _, outputs_u12 = net(inputs_u2)
            _, _, outputs_u21 = net2(inputs_u)
            _, _, outputs_u22 = net2(inputs_u2)  
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) 
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()       
            map_x, _, outputs_x = net(inputs_x, return_map=True)
            _, _, outputs_x2= net(inputs_x2)
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)
            targets_x =  ptx / ptx.sum(dim=1, keepdim=True)         
            targets_x =  targets_x.detach()       

        inputs_erase = torch.cat([inputs_x, inputs_u], dim=0)
        inputs_map = torch.cat([map_x, map_u], dim=0)
        gmm_weight = torch.cat([w_x, w_u], dim=0)
        target_mask = torch.cat([targets_x, targets_u], dim=0)
        erase_imgx_max, erase_imgx_min, target_mask = attention_erase_map(inputs_erase, inputs_map, gmm_weight, target_mask)
        targets_x_mask, targets_u_mask = target_mask[:batch_size], target_mask[batch_size:]

        reconimg = torch.cat([erase_imgx_max, erase_imgx_min], dim=0)
        reconout = net(reconimg, recon=True)
        Lrecon = F.mse_loss(reconout[:inputs_erase.shape[0]], inputs_erase, size_average=True) + F.mse_loss(reconout[inputs_erase.shape[0]:], inputs_erase, size_average=True)

        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        all_inputs = torch.cat([inputs_xp, inputs_xp2, inputs_xs1,inputs_xs2, erase_imgx_max[:batch_size], erase_imgx_min[:batch_size], inputs_up, inputs_up2, inputs_us1,inputs_us2, erase_imgx_max[batch_size:], erase_imgx_min[batch_size:]],dim=0)
        all_targets = torch.cat([targets_x, targets_x,  targets_x, targets_x, targets_x_mask, targets_x_mask, targets_u, targets_u, targets_u, targets_u, targets_u_mask, targets_u_mask],dim=0)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        _, features_mix, logits = net(mixed_input)
        logits_x = logits[:batch_size*6]
        logits_u = logits[batch_size*6:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*6], logits_u, mixed_target[batch_size*6:], epoch+batch_idx/num_iter, warm_up)

        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        print(Lx, Lu, penalty, Lrecon)
        loss = Lx + lamb * Lu  + penalty + Lrecon*0.02
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, _, outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step() 

def test(epoch,net1,net2, maxacc):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, _, outputs1 = net1(inputs)
            _, _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                           
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    maxacc=max(acc, maxacc)
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    print("max accuracy:", maxacc)
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  
    return maxacc

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            _, _, outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9:
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def create_model_selfsup(net='resnet18', dataset='cifar10', num_classes=10, device='cuda:0', drop=0, DivideMix=False):
    if DivideMix:
        model = ResNet18(num_classes=num_classes)
    else:
        chekpoint = torch.load('./pretrained/ckpt_{}_{}.pth'.format(dataset, net))
        sd = {}
        for ke in chekpoint['model']:
            nk = ke.replace('module.', '')
            sd[nk] = chekpoint['model'][ke]
        model = SupCEResNet(net, num_classes=num_classes)
        model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return model


def create_model(dataset='cifar10', num_classes=10):
    model = create_model_selfsup(dataset = dataset, num_classes = num_classes)
    model = model.cuda()
    return model

def create_model_noself(num_classes = num_classes):
    model = ResNet18(num_classes=num_classes)
    model = model.cuda()
    return model


checkpath=args.check_path
stats_log=open('./check_cifar-10'+'/checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./check_cifar-10'+'/checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
    num_classes = 10
elif args.dataset=='cifar100':
    warm_up = 20
    num_classes = 100

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
# net1 = create_model(dataset = args.dataset, num_classes= num_classes)
# net2 = create_model(dataset = args.dataset, num_classes= num_classes)

net1 = create_model(dataset='cifar10',num_classes= num_classes)
net2 = create_model(dataset='cifar10',num_classes= num_classes)

cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] 
maxacc=0
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) 
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)   
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) 
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)       

    maxacc=test(epoch,net1,net2,maxacc)  


