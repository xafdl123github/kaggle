import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
import torch.nn.functional as F
from icecream import ic
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import itertools
import torchvision.transforms as transforms
from PIL import Image
from utils import RandomRotation, RandomShift
from colorama import Style, Fore, Back

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
train_raw = pd.read_csv('./train.csv')   # (42000, 785)
test_raw = pd.read_csv('./test.csv')    # (28000, 784)

# 划分数据
split_ratio = 0.9
train_ini = train_raw[:int(split_ratio * train_raw.shape[0])]   # (33600, 785)
valid_ini = train_raw[int(split_ratio * train_raw.shape[0]):]   # (8400, 785)

y = train_ini['label']  # 目标
y_valid = valid_ini['label']  # 目标

# 最终的训练数据, 验证数据
train = train_ini.drop('label', axis=1)
valid = valid_ini.drop('label', axis=1)

# 映射到0-1之间
# train = train.values / 255.0
# valid = valid.values / 255.0
train = train.values  # (33600, 784)
valid = valid.values
test = test_raw.values

train = train.reshape(-1,1,28,28).astype(np.float32)    # 如果不转换类型会出现 RuntimeError: expected scalar type Double but found Float 错误  train.shape: (42000, 1, 28, 28)
valid = valid.reshape(-1,1,28,28).astype(np.float32)    # 如果不转换类型会出现 RuntimeError: expected scalar type Double but found Float 错误  train.shape: (42000, 1, 28, 28)
test = test.reshape(-1,1,28,28).astype(np.float32)    # 如果不转换类型会出现 RuntimeError: expected scalar type Double but found Float 错误  train.shape: (42000, 1, 28, 28)

# 转换为numpy
y = y.values
y_valid = y_valid.values
y_test = np.zeros(test.shape[0])

# 定义数据集
class MnistDataset(object):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        target = self.y[idx]
        img = self.X[idx]

        if self.transform is not None:
            img = np.transpose(img, (1, 2, 0))
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.X)


"""数据增强"""
transform_train = transforms.Compose([
    # transforms.RandomCrop(20),
    # transforms.ColorJitter(brightness=1, contrast=0.1, hue=0.5),
    transforms.ToPILImage(),
    # transforms.RandomRotation(5),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomCrop(size=28, padding=6),  # 几乎没有效果，收敛有点慢
    RandomRotation(degrees=20),  # 角度过小时(10,15)，容易出现过拟合
    RandomShift(3),
    # transforms.GaussianBlur(1, (1,4)),  # 有问题，收敛极慢
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),

])
transform_valid = transforms.Compose([
    transforms.ToPILImage(),
    # RandomRotation(degrees=15),
    # RandomShift(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# 实例化 将数据集组合为dataset
ds = MnistDataset(train, y, transform_train)
ds_valid = MnistDataset(valid, y_valid, transform_valid)
ds_test = MnistDataset(test, y_test, transform_valid)

# 转换成DataLoader
valid_loader = DataLoader(ds_valid, batch_size=64, num_workers=0, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=64, num_workers=0, shuffle=False)

# 定义网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        """
        kernel=5 要比 kernel=3 收敛速度更快，准确率更高
        kernel=7 已经出现过拟合了
        padding越大，好像越容易出现过拟合，模型的学习能力还是很强的
        """

        self.conv_unit = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, stride=2),

        )

        self.fc_unit = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x, train=True):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        x = self.fc_unit(x)
        return x

# 实例化
model = CNN()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 转移到相应设备
model = model.to(device)

# 定义损失函数，分类问题用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器 小批量随机梯度下降法
# optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

since_time = time.time()

train_acc = []
train_loss = []

# 超参数
EPOCH = 50

color_dict = {'train': Fore.WHITE, 'valid': Fore.GREEN}

loss_dict = {'train': [], 'valid': []}

# 开始训练  +++++++++++++++++++++
for epoch in range(1, EPOCH + 1):

    train_loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=True)

    loader_dict = {'train': train_loader, 'valid': valid_loader}

    print('Epoch %d/%d' % (epoch, EPOCH))
    print('-'*10)

    for phase, loader in loader_dict.items():

        running_loss = 0.0
        running_corrects = 0

        for i, data in enumerate(loader, 0):

            images = data[0]    # torch.Size([8, 1, 28, 28])
            labels = data[1]    # torch.Size([8])

            # 转移到相应设备
            images = images.to(device)
            labels = labels.to(device)

            # 模型参数的梯度清零
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # 训练
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, dim=1)

                if phase == 'train':
                    # 反向传播：计算模型参数的梯度；更新模型的参数
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum()

        print('%s  Loss：%.3f  Acc: %.2f %%' % (phase, running_loss / len(loader.dataset), running_corrects / len(loader.dataset) * 100))
        loss_dict[phase].append(running_loss / len(loader.dataset))

    # 一个Epoch结束后加一个空行
    print()

total_sec = time.time() - since_time
print('总共耗时：%d分 %d秒' % (total_sec // 60, total_sec % 60))

# 绘图
plt.plot(range(len(loss_dict['train'])), loss_dict['train'], c='red', label='train loss')
plt.plot(range(len(loss_dict['valid'])), loss_dict['valid'], c='green', label='valid loss')
plt.title('训练模型', fontsize=20)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

# exit()

"""评估模型"""

# 测试集上整体的准确率
correct = 0
total = 0

total_lst = list(range(10))
corr_lst = list(range(10))

# 不需要追踪历史
pred_container = []
label_container = []

with torch.no_grad():

    for i, data in enumerate(valid_loader):
        # data[0].shape: torch.Size([64, 1, 28, 28])
        # data[1].shape: torch.Size([64])

        images = data[0].to(device)
        labels = data[1].to(device)

        # 预测
        outputs = model(images, train=False)

        named_tuple = torch.max(outputs, dim=1)

        # 预测值
        predicted = named_tuple[1]

        # 将预测值转为List，然后添加到容器中
        for n in predicted.tolist():
            pred_container.append(n)
        for n in labels.tolist():
            label_container.append(n)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for k, pred in enumerate(predicted):
            total_lst[labels[k]] += 1
            # 预测正确
            if labels[k] == pred:
                corr_lst[pred] += 1


print('Accuracy of valid data is %.3f %%' % (correct / total * 100))

print('\n每个类别的准确率')

# 计算每个类别的准确率
lst = list()
for i, num in enumerate(corr_lst):
    lst.append(num / total_lst[i] * 100)

# 排序
lst = np.array(lst)
sorted_index = np.argsort(-lst)
for k, v in enumerate(sorted_index):
    print('Accuracy of %d：%.1f %%' % (v, lst[v]))

# exit()

# 将预测结果转换成numpy
pred_container = np.array(pred_container)
label_container = np.array(label_container)
# ic(type(pred_container), pred_container, pred_container.shape)
# ic(type(label_container), label_container, label_container.shape)

# 计算混淆矩阵
confusion_mtx = confusion_matrix(label_container, pred_container)

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color='red')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, classes=range(10))


"""生成csv文件"""
test_container = []
with torch.no_grad():

    for i, data in enumerate(test_loader):
        # data[0].shape: torch.Size([64, 1, 28, 28])
        # data[1].shape: torch.Size([64])

        images = data[0].to(device)
        # labels = data[1].to(device)

        # 预测
        outputs = model(images)

        named_tuple = torch.max(outputs, dim=1)

        # 预测值
        predicted = named_tuple[1]

        # 将预测值转为List，然后添加到容器中
        for n in predicted.tolist():
            test_container.append(n)

# 将预测结果转换成numpy
test_container = np.array(test_container)
ImageId = np.arange(1, test_container.shape[0] + 1)
submission = pd.DataFrame({'ImageId': ImageId, 'Label': test_container})
submission.to_csv('./submit/sub3.csv', index=False)

