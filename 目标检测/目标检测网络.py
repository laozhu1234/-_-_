import torch
import time
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import os
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


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_loss = 100000.0
# print(device)

#搭建VGG-11
class VGG_Net(nn.Module):
    def __init__(self):
        super(VGG_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3,padding=1)
        self.conv33 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv44 = nn.Conv2d(256, 256, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(7*7*256, 1024)
        self.fc3 = nn.Linear(1024, 22)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv33(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv44(x))
        x = self.pool(x)

        x = F.relu(self.conv44(x))
        x = F.relu(self.conv44(x))
        x = self.pool(x)


        x = x.view(-1, 256*7*7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


def training(net,img_train,centers_train,img_test,centers_test,criterion,optimizer,net_path,batches):
    print("start training")
    for epoch in range(200):
        s = time.time()
        #一批一批来，30个一批
        for i in range(0, img_train.shape[0], batches):
            x = img_train[i:i + batches]
            y = centers_train[i:i + batches]
            net.train()
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(y, outputs)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                net.eval()
                predict = net(img_test)
                test_loss = criterion(predict, centers_test)#计算出测试误差
                print("[%d, %d] training_loss:%0.4f testing_loss:%0.4f" % (
                epoch + 1, i + batches, loss * 2, test_loss * 2))
            global best_loss#best_loss为全局变量
            #test_loss为平均一个坐标值的误差，test_loss * 2为平均一对坐标值的误差
            if (test_loss * 2 < best_loss):
                torch.save(net, net_path)
                best_loss = test_loss * 2
        e = time.time()
        print("epoch:%d time:%f" %(epoch+1, e-s))
    print("finish training")

#计算测试数据的平均一对坐标的误差
def testing(cen_org_test, outputs_cen, scale_test, criterion):
    with torch.no_grad():
        cen_org_test = torch.from_numpy(cen_org_test)
        cen_org_test = cen_org_test.to(torch.float32)
        for i in range(outputs_cen.shape[0]):
            for j in range(22):
                if j%2 == 0:
                    outputs_cen[i][j] /= scale_test[i][0]
                else:
                    outputs_cen[i][j] /= scale_test[i][1]
        test_loss = criterion(cen_org_test, outputs_cen)
        # test_loss为平均一个坐标值的误差，test_loss * 2为平均一对坐标值的误差
        print('testing loss:%f' %(math.sqrt(test_loss.item()*2)))

if __name__ == '__main__':
    #导入训练数据
    img_train = np.load('img_train.npy')
    centers_train = np.load('center_train.npy')

    img_train = torch.from_numpy(img_train)
    centers_train = torch.from_numpy(centers_train)
    img_train = img_train.to(torch.float32)
    centers_train = centers_train.to(torch.float32)
    img_train = img_train.permute(0, 3, 1, 2)
    centers_train = centers_train.view(-1, 22)
    #导入测试数据
    img_test = np.load('img_test.npy')
    centers_test = np.load('center_test.npy')

    img_test = torch.from_numpy(img_test)
    centers_test = torch.from_numpy(centers_test)
    img_test = img_test.to(torch.float32)
    centers_test = centers_test.to(torch.float32)
    img_test = img_test.permute(0, 3, 1, 2)
    centers_test = centers_test.view(-1, 22)
    #训练VGG
    batches = 30
    net_path = 'center_VGG_best.pth'#已经训练好的网络
    net = VGG_Net()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)#优化器
    criterion = nn.MSELoss(reduction='mean')#均方误差

    #运行训练部分的代码前，先将下面注释去掉
    # net_path_train = 'center_VGG.pth'
    # training(net,img_train,centers_train,img_test,centers_test,criterion,optimizer,net_path_train,batches)

    # 测试部分的代码
    cen_org_test = np.load('center_org_test.npy')
    cen_org_test = np.reshape(cen_org_test,(-1,22))
    scale_test = np.load('scale_test.npy')
    net = torch.load(net_path)
    net.eval()
    with torch.no_grad():      
        outputs = net(img_test)
        np.save('center_test_predicted.npy',outputs.detach().numpy()) #保存预测结果,(51*22)，用于可视化   
        testing(cen_org_test, outputs, scale_test, criterion)






