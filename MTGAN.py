import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
import sys
from sklearn.decomposition import PCA

# setting parameters
DataPath = '/home/zf/PycharmProjects/untitled/HSI_Hang/IndinePines2010/Indian_pines_corrected.mat'
TRPath = '/home/zf/PycharmProjects/untitled/HSI_Hang/IndinePines2010/TRLabel.mat'
TSPath = '/home/zf/PycharmProjects/untitled/HSI_Hang/IndinePines2010/TSLabel.mat'
savepath = '/home/zf/PycharmProjects/untitled/HSI_Hang/IndinePines2010/'+str(sys.argv[1])+'.mat'

patchsize = 16  # input spatial size for 2D-CNN
batchsize = 256  # select from [16, 32, 64, 128], the best is 64
EPOCH = 100
LR = 0.001

# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)

Data = Data['indian_pines_corrected']
Data = Data.astype(np.float32)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']


pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
[m, n, l] = np.shape(Data)

#data pre-processing1
for i in range(l):
    Data[:, :, i] = (Data[:, :, i]-Data[:, :, i].min()) / (Data[:, :, i].max()-Data[:, :, i].min())
x = Data

#data pre-processing2
# for i in range(l):
#     mean = np.mean(Data[:, :, i])
#     std = np.std(Data[:, :, i])
#     Data[:, :, i] = (Data[:, :, i] - mean)/std
#     # x2[:, :, i] = np.pad(Data[:, :, i], pad_width, 'symmetric')

# extract the first principal component
# x = np.reshape(Data, (m*n, l))
# pca = PCA(n_components=200, copy=True, whiten=False)
# x = pca.fit_transform(x)
# _, l = x.shape
# x = np.reshape(x, (m, n, l))
# print(x.shape)
# plt.figure()
# plt.imshow(x)
# plt.show()

# boundary interpolation
temp = x[:, :, 0]
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
x2 = np.empty((m2, n2, l), dtype='float32')

for i in range(l):
    temp = x[:, :, i]
    pad_width = np.floor(patchsize/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x2[:, :, i] = temp2

# construct the training and testing set
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, l, patchsize, patchsize), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
    patch = np.reshape(patch, (patchsize * patchsize, l))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (l, patchsize, patchsize))
    TrainPatch[i, :, :, :] = patch
    patchlabel = TrLabel[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch = np.empty((TestNum, l, patchsize, patchsize), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
    patch = np.reshape(patch, (patchsize * patchsize, l))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (l, patchsize, patchsize))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel

print('Training size and testing size are:', TrainPatch.shape, 'and', TestPatch.shape)

# step3: change data to the input type of PyTorch
TrainPatch = torch.from_numpy(TrainPatch)
TrainLabel = torch.from_numpy(TrainLabel)-1
TrainLabel = TrainLabel.long()
dataset = dataf.TensorDataset(TrainPatch, TrainLabel)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

TestPatch = torch.from_numpy(TestPatch)
TestLabel = torch.from_numpy(TestLabel)-1
TestLabel = TestLabel.long()

Classes = len(np.unique(TrainLabel))

OutChannel = 32


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=l,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.dconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )

        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, l, 3, 1, 1),
            nn.BatchNorm2d(l),
            nn.Sigmoid(),

        )

    def forward(self, x, x1, x2):
        x = self.dconv1(x)
        x = torch.cat([x, x1], 1)
        x = self.dconv2(x)
        x = torch.cat([x, x2], 1)
        x = self.dconv3(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # nn.Dropout(0.5),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # nn.Dropout(0.5),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # nn.Dropout(0.5),

        )

        self.out1 = nn.Linear(256, Classes)  # fully connected layer, output 16 classes
        self.out2 = nn.Linear(256, Classes)
        self.out3 = nn.Linear(256, Classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.33]))

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x1 = x1.view(x1.size(0), -1)
        # x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.out1(x1)
        x1 = F.softmax(x1)
        x2 = x2.view(x2.size(0), -1)
        # x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.out2(x2)
        x2 = F.softmax(x2)
        x3 = x3.view(x3.size(0), -1)
        # x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.out3(x3)
        x3 = F.softmax(x3)
        out = x1 * self.coefficient1 + x2 * self.coefficient2 + x3 * self.coefficient3
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()

    def forward(self, x):
        ex1, ex2, ex3 = self.encoder(x)
        rx = self.decoder(ex3, ex2, ex1)
        cx = self.classifier(ex3, ex2, ex1)
        rx = rx.view(rx.size(0), -1)
        return rx, cx


cnn = Network()
dis = Discriminator()
print('The structure of the designed network', cnn)

# display variable name and shape
# for param_tensor in cnn.state_dict():
#     print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())

# load pre-trained ae parameters
model_dict = cnn.state_dict()
pretrained_dict = torch.load('net_params_MTCNN_HS.pkl')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
cnn.load_state_dict(model_dict)

# move model to GPU
cnn.cuda()
dis.cuda()

g_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
d_optimizer = torch.optim.Adam(dis.parameters(), lr=LR)
loss_fun1 = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_fun2 = nn.MSELoss()

BestAcc = 0
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

        # move train data to GPU
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        fake_img, output = cnn(b_x)  # cnn output

        dis.zero_grad()
        fake = dis(fake_img.view(fake_img.size(0), l, patchsize, patchsize)).mean()
        real = dis(b_x).mean()
        d_loss = 1 - real + fake
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        cnn.zero_grad()
        ce_loss = loss_fun1(output, b_y)
        a_loss = torch.mean(1 - fake)
        g_loss = 0.01 * a_loss + ce_loss + 1 * F.binary_cross_entropy(fake_img, b_x.view(b_x.size(0), -1))
        g_loss.backward()
        g_optimizer.step()

        if step % 50 == 0:
            cnn.eval()

            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 100
            for i in range(number):
                temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
                temp = temp.cuda()
                _, temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                del temp, _, temp2, temp3

            if (i + 1) * 100 < len(TestLabel):
                temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
                temp = temp.cuda()
                _, temp2 = cnn(temp)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                del temp, _, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
            # test_output = rnn(TestData)
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            # accuracy = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)
            print('Epoch: ', epoch, '| classify loss: %.6f' % ce_loss.data.cpu().numpy(), '| adversarial loss: %.6f' % d_loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'net_params_AMTCNN_HS.pkl')
                BestAcc = accuracy

            cnn.train()


# # test each class accuracy
# # divide test set into many subsets

cnn.load_state_dict(torch.load('net_params_AMTCNN_HS.pkl'))
cnn.eval()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//100
for i in range(number):
    temp = TestPatch[i*100:(i+1)*100, :, :]
    temp = temp.cuda()
    _, temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*100:(i+1)*100] = temp3.cpu()
    del temp, _, temp2, temp3

if (i+1)*100 < len(TestLabel):
    temp = TestPatch[(i+1)*100:len(TestLabel), :, :]
    temp = temp.cuda()
    _, temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*100:len(TestLabel)] = temp3.cpu()
    del temp, _, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

Classes = np.unique(TestLabel)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel)):
        if TestLabel[j] == cla:
            sum += 1
        if TestLabel[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()


print(OA)
print(EachAcc)

del TestPatch, TrainPatch, TrainLabel, b_x, b_y, dataset, train_loader
# show the whole image
# The whole data is too big to test in one time; So dividing it into several parts
part = 100
pred_all = np.empty((m*n, 1), dtype='float32')

number = m*n//part
for i in range(number):
    D = np.empty((part, l, patchsize, patchsize), dtype='float32')
    count = 0
    for j in range(i*part, (i+1)*part):
        row = j//n
        col = j - row*n
        row2 = row + pad_width
        col2 = col + pad_width
        patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        D[count, :, :, :] = patch
        count += 1

    temp = torch.from_numpy(D)
    temp = temp.cuda()
    _, temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[i*part:(i+1)*part, 0] = temp3.cpu()
    del temp, _, temp2, temp3, D

if (i+1)*part < m*n:
    D = np.empty((m*n-(i+1)*part, l, patchsize, patchsize), dtype='float32')
    count = 0
    for j in range((i+1)*part, m*n):
        row = j // n
        col = j - row * n
        row2 = row + pad_width
        col2 = col + pad_width
        patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        D[count, :, :, :] = patch
        count += 1

    temp = torch.from_numpy(D)
    temp = temp.cuda()
    _, temp2 = cnn(temp)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_all[(i + 1) * part:m*n, 0] = temp3.cpu()
    del temp, _, temp2, temp3, D


pred_all = np.reshape(pred_all, (m, n)) + 1
OA = OA.numpy()
pred_y = pred_y.cpu()
pred_y = pred_y.numpy()
TestDataLabel = TestLabel.cpu()
TestDataLabel = TestDataLabel.numpy()

io.savemat(savepath, {'PredAll': pred_all, 'OA': OA, 'TestPre': pred_y, 'TestLabel': TestDataLabel})
print('Save mat file')

plt.figure()
plt.imshow(pred_all)
plt.show()