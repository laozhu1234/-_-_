import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Resnet18_2 import ResNet18
from Resnet18_2 import ResidualBlock
from Resnet18_2 import ResNet_18
from sklearn import metrics

net18_2 = torch.load('./model/Resnet18_2.pth')

images = np.load("./Resnet18_2/test_img.npy")
images = torch.from_numpy(images)
labels = np.load("./Resnet18_2/test_label.npy")
labels = torch.from_numpy(labels)

correct = 0
total = 0
one = 0

net18_2.eval()
with torch.no_grad():
    images, labels = images.to(torch.float32), labels.to(torch.long)
    outputs = net18_2(images)
    # 取得分最高的那个类 (outputs.data的索引号)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    cfm = metrics.confusion_matrix(labels, predicted)
    correct += (predicted == labels).sum()
    one += (predicted == 1).sum()
    print("———————————————————————TEST STAT———————————————————————")
    print('混淆矩阵:\n', cfm)
    print('test acc: %.2f%%'
          % (100. * correct / total))
    print("———————————————————————TEST  END———————————————————————")
    acc = 100. * correct / total


