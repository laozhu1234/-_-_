import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn import metrics


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet_18():
    return ResNet18(ResidualBlock)


def train(net, inputs, labels, optimizer, EPOCH, criterion):
    net.train()
    correct = 0.0
    avg_loss = 0.0
    total = 0.0
    inputs, labels = inputs.to(torch.float32), labels.to(torch.long)
    bantch = [0, 200, 400, 600, 736]

    iter_num = 4
    for i in range(iter_num):
        s = time.time()
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs[bantch[i]:bantch[i + 1]])
        loss = criterion(outputs, labels[bantch[i]:bantch[i + 1]])
        loss.backward()
        optimizer.step()

        # 每训练1个epoch打印一次loss和准确率
        loss = loss.item()
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels[bantch[i]:bantch[i + 1]].size(0)
        correct += predicted.eq(labels[bantch[i]:bantch[i + 1]].data).cpu().sum()

        e = time.time()
        print('[epoch:%d | iter:%d] Loss: %.03f | Acc: %.2f%% %0.2fs'
              % (EPOCH + 1, i + 1, loss, 100. * correct / total, (e - s)))

    avg_loss /= iter_num
    print("[epoch:%d] Avg loss: %.03f" % (EPOCH + 1, avg_loss))


def tst(net, images, labels, best_acc, criterion):
    with torch.no_grad():
        correct = 0
        total = 0
        one = 0
        zero = 0

        net.eval()
        images, labels = images.to(torch.float32), labels.to(torch.long)
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss = loss.item()
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        cfm = metrics.confusion_matrix(labels, predicted)
        correct += (predicted == labels).sum()
        one += (predicted == 1).sum()
        zero += (predicted == 0).sum()
        print("———————————————————————TEST STAT———————————————————————")
        print('混淆矩阵:\n', cfm)
        print('test loss: %.03f | test acc: %.2f%%'
              % (loss, 100. * correct / total))
        print("———————————————————————TEST  END———————————————————————")
        acc = 100. * correct / total
        if acc >= best_acc and (one / total) <= 0.95 and (zero / total) <= 0.95:
            best_acc = acc
            torch.save(net, './model/Resnet18_2.pth')
    return best_acc


def main():
    s_t = time.time()
    LR = 0.001  # 学习率
    torch.manual_seed(0)

    train_inputs = np.load("./Resnet18_2/train_img.npy")
    train_inputs = torch.from_numpy(train_inputs)
    train_labels = np.load("./Resnet18_2/train_label.npy")
    train_labels = torch.from_numpy(train_labels)

    test_images = np.load("./Resnet18_2/test_img.npy")
    test_images = torch.from_numpy(test_images)
    test_labels = np.load("./Resnet18_2/test_label.npy")
    test_labels = torch.from_numpy(test_labels)

    # test_images = train_inputs
    # test_labels = train_labels

    # 模型定义-ResNet
    net = ResNet_18().to()

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.5)  # 优化方式为mini-batch
    # optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.8, 0.999),
    #                        eps=1e-08,
    #                        weight_decay=0.001,
    #                        amsgrad=False)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    EPOCH = 20  # 遍历数据集次数
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, Resnet-18!")
    for epoch in range(EPOCH):
        train(net, train_inputs, train_labels, optimizer, epoch, criterion)
        best_acc = tst(net, test_images, test_labels, best_acc, criterion)
    e_t = time.time()
    print("———————————————————————Finish———————————————————————")
    print("best_acc =" + str(best_acc) + " | Total time =" + str(e_t - s_t) + "s")


if __name__ == '__main__':
    main()
