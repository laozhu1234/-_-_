import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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
import time
import pickle
import glob
import io
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pre_rule = 10

#搭建VGG
class VGG_Net(nn.Module):
    def __init__(self):
        super(VGG_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3,padding=1)
        self.conv11 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3,padding=1)
        self.conv22 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv33 = nn.Conv2d(256, 256, 3,padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3,padding=1)
        self.conv44 = nn.Conv2d(512, 512, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(p = 0.5)
        # self.conv1128 = nn.Conv2d(128, 128, 1)
        # self.conv1256 = nn.Conv2d(256, 256, 1)
        # self.conv1512 = nn.Conv2d(512, 512, 1)
        # self.conv128to64 = nn.Conv2d(128, 64, 1)
        # self.conv64to128 = nn.Conv2d(64, 128, 3,padding=1)
        # self.conv128to256= nn.Conv2d(128, 256, 3,padding=1)
        # self.conv256to128 = nn.Conv2d(256, 128, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv11(x))
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn1(self.conv11(x)))
        x = self.pool(x)

        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv22(x))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn2(self.conv22(x)))
        x = self.pool(x)

        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv33(x))
        # x = F.relu(self.conv33(x))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn3(self.conv33(x)))
        # x = F.relu(self.bn3(self.conv33(x)))
        # x = F.relu(self.conv1256(x))
        x = self.pool(x)

        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv1256(x))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn4(self.conv44(x)))
        # x = F.relu(self.bn4(self.conv44(x)))
        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv1512(x))
        x = self.pool(x)

        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv44(x))
        # x = F.relu(self.conv1256(x))
        # x = F.relu(self.bn4(self.conv44(x)))
        x = F.relu(self.bn4(self.conv44(x)))
        # x = F.relu(self.bn4(self.conv44(x)))
        # x = F.relu(self.conv1512(x))
        x = self.pool(x)

        x = x.view(-1, 512)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def training(net,train_loader,test_loader,criterion,optimizer,net_path,batches):
    print('start Training')
    for epoch in range(10):
        s = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            net.train()
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1)*batches % 160 == 0:
                print('[{}, {}] loss: {:.3f}' .format(epoch + 1, (i + 1)*batches, running_loss / (160/batches)))
                running_loss = 0.0
                net.eval()
                testing_select_model(test_loader, net, criterion, net_path)
        e = time.time()
        print("epoch:%d time:%f" % (epoch + 1, e - s))
    print('Finished Training')

def testing_select_model(test_loader, net, criterion, net_path):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1]
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct/total
    print('testing_accuracy:{}% ,testing_loss:{}'.format(acc * 100, loss))
    # 选模型时，采用验证集的错误率和交叉熵的加权和最小化的方式
    global pre_rule
    new_rule = 0.5 * loss + 0.5 * (1 - acc)
    if new_rule < pre_rule:
        torch.save(net, net_path)
        pre_rule = new_rule

def testing_print_acc(test_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1]
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct/total
    print('testing_accuracy:{}% ,testing_loss:{}'.format(acc * 100, loss))

if __name__ == '__main__':
    my_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batches = 80
    train_dataset = datasets.ImageFolder(root='./new train', transform=my_transform)
    train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=1)
    test_dataset = datasets.ImageFolder(root='./test', transform=my_transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset.targets), shuffle=True, num_workers=1)
    valid_test_dataset = datasets.ImageFolder(root='./predicted test_5', transform=my_transform)
    valid_test_loader = DataLoader(valid_test_dataset, batch_size=len(valid_test_dataset.targets), shuffle=True,num_workers=1)
    #训练VGG
    net = VGG_Net()
    criterion = nn.CrossEntropyLoss()  # 交叉熵
    optimizer = optim.Adam(net.parameters(), lr=0.0002)#优化器

    # 若要运行训练部分的代码，先将下面注释去掉
    # net_path_train = 'new 2 classes_VGG.pth'
    # training(net,train_loader,test_loader,criterion,optimizer,net_path_train,batches)

    # 测试部分的代码
    net.eval()
    net_path_test = 'new 5 classes_VGG_best.pth'
    mynet = torch.load(net_path_test)
    testing_print_acc(valid_test_loader, mynet)



