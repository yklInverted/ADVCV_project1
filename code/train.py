from deepfashion import DeepFashion
from model import ResNet34, ResNet50, ResNet101, MultiLabelResNet
from utils import *

import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

import numpy as np
import os

# Training
def train(epoch, net, criterion, trainloader, scheduler, optimizer):
    device = 'cuda'
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_weight = [0.22902098, 0.1215035 , 0.11625874, 0.19318182, 0.20629371,
       0.13374126]
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        #ipdb.set_trace()
        outputs = net(inputs)
        loss = 0
        
        i = 0
        start = 0
        
        for num in net.class_num:
            loss += loss_weight[i] * criterion(outputs[:, start: start + num], targets[:, i])
            
            _, predicted = outputs[:, start: start + num].max(1)
            correct += predicted.eq(targets[:,i]).sum().item()
            start += num
            i += 1
            
        #loss = loss/len(net.class_num)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #_, predicted = outputs.max(1)
        total += targets.size(0)*targets.size(1)
        #correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 50 == 0:
          print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (batch_idx+1, train_loss/(batch_idx+1), 100.*correct/total))

    scheduler.step()
    return train_loss/(batch_idx+1), 100.*correct/total

def test(net, criterion, testloader):
    device = 'cuda'
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = 0
        
            i = 0
            start = 0
            for num in net.class_num:
                loss += criterion(outputs[:, start: start + num], targets[:, i])
                _, predicted = outputs[:, start: start + num].max(1)
                correct += predicted.eq(targets[:,i]).sum().item()
                start += num
                i += 1
                
            valid_loss += loss.item()
            #_, predicted = outputs.max(1)
            total += targets.size(0)*targets.size(1)
            #correct += predicted.eq(targets).sum().item()

    return valid_loss/(batch_idx+1), 100.*correct/total

def generate_output(net, testloader):
    device = 'cuda'
    net.eval()
    result = []
    
    with torch.inference_mode():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            result_batch = torch.zeros(outputs.shape[0], len(net.class_num)).to(device)
        
            i = 0
            start = 0
            for num in net.class_num:
                
                _, predicted = outputs[:, start: start + num].max(1)
                result_batch[:,i] = predicted
                start += num
                i += 1
                
            result.append(result_batch)
    
    result = torch.concat(result, 0)
    write_txt(result.cpu().numpy(), 'result.txt')
    return 0

if __name__=="__main__":
    
    parser = config_parser()
    args = parser.parse_args()
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.76575, 0.7359, 0.7254), (0.2850, 0.2978, 0.3039)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.76575, 0.7359, 0.7254), (0.2850, 0.2978, 0.3039)),
    ])
    
    #create dataloader
    trainset = DeepFashion('../FashionDataset','train', transform= transform_train)
    valset = DeepFashion('../FashionDataset','val', transform = transform_test)
    CLASS_NUM = [7,3,3,4,6,3]
    
    #GT Labels of testset are not given, cannot test, only for evaluation
    testset = DeepFashion('../FashionDataset','test', transform = transform_test)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    validloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    
    # main body
    config = {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-3
    }

    train_loss_ls = []
    train_acc_ls = []
    valid_loss_ls = []
    valid_acc_ls = []

    net = MultiLabelResNet(class_num = CLASS_NUM).to('cuda')
    if os.path.isfile('checkpoint/ckpt.pth') and args.load_checkpoint:
        state = torch.load('checkpoint/ckpt.pth')
        net.load_state_dict(state['net'])
        start_epoch = state['epoch']
    else:
        start_epoch = 0
    
    i = 0
    for p in net.resnet.parameters():
        p.requires_grad = False
        #print(p.size())
        i += 1
        if i == args.freeze_layer:
            break
    
    print(f'Total parameters are {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), lr=config['lr'],
                        momentum=config['momentum'], weight_decay=config['weight_decay'])
    #optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    for epoch in range(start_epoch, args.epoch):
        train_loss, train_acc = train(epoch, net, criterion, trainloader, scheduler, optimizer)
        valid_loss, valid_acc = test(net, criterion, validloader)
        
        print(("Epoch : %3d, training loss : %0.4f, training loss : %2.2f, validation loss " + \
        ": %0.4f, validation loss : %2.2f") % (epoch, train_loss, train_acc, valid_loss, valid_acc))

        train_loss_ls.append(train_loss)
        valid_loss_ls.append(valid_loss)
        train_acc_ls.append(train_acc)
        valid_acc_ls.append(valid_acc) 

        #Finally, get the test result
        #test_loss, test_acc = test(epoch, net, criterion, testloader)
    ipdb.set_trace()
    save_checkpoint(net, valid_acc, epoch)
    generate_output(net, testloader)