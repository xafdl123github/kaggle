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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
train_raw = pd.read_csv('./train.csv')   # (42000, 785)
test_raw = pd.read_csv('./test.csv')    # (28000, 784)

# 划分数据
split_ratio = 0.8
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

# 超参数
EPOCH = 15

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
            # # 去掉第一维
            # img_drop0 = np.squeeze(img, 0)
            # img2 = Image.fromarray(img_drop0, mode='L')  # 将numpy转换为Image类型
            # img2 = self.transform(img2)  # transform接收的参数必需是Image类型
            # ret_img = np.array(img2)
            # # 增加第一维
            # img = np.expand_dims(ret_img, 0)
            # img = img / 255.0
            # img = np.array(img, dtype=np.float32)  # 转换为float类型

        return img, target

    def __len__(self):
        return len(self.X)


"""数据增强"""
transform = transforms.Compose([
    # transforms.RandomCrop(20),
    # transforms.RandomRotation(5),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=1, contrast=0.1, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# 实例化 将数据集组合为dataset
ds = MnistDataset(train, y, transform)
ds_valid = MnistDataset(valid, y_valid, transform)
ds_test = MnistDataset(test, y_test, transform)

# 转换成DataLoader
train_loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=True)
valid_loader = DataLoader(ds_valid, batch_size=64, num_workers=0, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=64, num_workers=0, shuffle=False)

# 定义网络
class CNN(nn.Module):

    def __init__(self):
        """
        (28 + 2*padding - kernel_size) / stride + 1
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # (8, 32, 28, 28)
        self.pool = nn.MaxPool2d((2, 2))  # (8, 32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (8, 64, 14, 14)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # ic| x.shape: torch.Size([8, 1, 28, 28])
        x = self.pool(F.relu(self.conv1(x)))  # (8, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (8, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (8, 10)
        return x

# 实例化
model = CNN()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 转移到相应设备
model = model.to(device)

# 定义损失函数，分类问题用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器 小批量随机梯度下降法
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

since_time = time.time()

train_acc = []
train_loss = []

# 开始训练
for epoch in range(EPOCH):


    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        images = data[0]    # torch.Size([8, 1, 28, 28])
        labels = data[1]    # torch.Size([8])

        # 转移到相应设备
        images = images.to(device)
        labels = labels.to(device)

        # ic(images.shape, labels.shape)
        # ic(images)

        # 训练
        outputs = model(images)
        # ic(outputs.shape)
        # ic(outputs)
        # ic(outputs[0])

        loss = criterion(outputs, labels)
        # ic(loss, type(loss))

        # 模型参数的梯度清零
        optimizer.zero_grad()

        # 反向传播：计算模型参数的梯度
        loss.backward()

        # 更新模型的参数
        optimizer.step()

        running_loss += loss.item()
        train_loss.append(loss.item())

        # 每100个batch打印一下损失
        if i % 100 == 99:
            print('[Epoch: %d, Batch: %d] Loss: %.3f' % (epoch, i + 1, running_loss / 100))
            running_loss = 0.0

        # 计算一个batch的准确率
        named_tuple = torch.max(outputs, dim=1)
        correct =  (named_tuple[1] == labels).sum().item()
        train_acc.append(correct / labels.size(0))

total_sec = time.time() - since_time
print('总共耗时：%d分 %d秒' % (total_sec // 60, total_sec % 60))

# 绘图
# plt.plot(range(len(train_loss)), train_loss, c='red', label='train loss')
# plt.plot(range(len(train_acc)), train_acc, c='green', label='train accuracy')
# plt.title('训练模型', fontsize=20)
# plt.xlabel('iter')
# plt.ylabel('score')
# plt.legend(loc='best')
# plt.show()

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
        outputs = model(images)

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


print('Accuracy of valid data is %.1f %%' % (correct / total * 100))

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
submission.to_csv('./submit/sub2.csv', index=False)

