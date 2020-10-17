import torch
from tqdm import tqdm
import os
import numpy as np
from model import Network
import torch.utils.data as data
from dataloader import Mahjong
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
print(1)
batch_size = 10

dataset = Mahjong()

#def weights_init(m):
#    classname=m.__class__.__name__
#    if classname.find('Conv') != -1:
#        xavier_uniform_(m.weight.data)
#        xavier_uniform_(m.bias.data)
net = Network()
def weights_init(m):    ##定义参数初始化函数                  
    classname = m.__class__.__name__    # m作为一个形参，原则上可以传递很多的内容，为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字。具体例子下边会详细说明。
    if classname.find('Conv') != -1:#find()函数，实现查找classname中是否含有conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)#m.weight.data表示需要初始化的权重。 nn.init.normal_()表示随机初始化采用正态分布，均值为0，标准差为0.02.
    elif classname.find('BatchNorm') != -1:           
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0) # nn.init.constant_()表示将偏差定义为常量0 
net.apply(weights_init)
net = net.double()
print(2)

dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, shuffle=True)
optim = torch.optim.Adam(net.parameters(), lr=0.1, betas=(0.5, 0.999))

fixed_x, fixed_label = dataset.get_fixed()
fixed_x =torch.tensor(fixed_x)
fixed_label = torch.tensor(fixed_label)
print(3)

for epoch in range(1):
    for i, data in tqdm(enumerate(dataLoader, 0)):
        net.zero_grad()
        x = Variable(data[0])
        label = Variable(data[1])
        y = net(x)
        loss = torch.nn.BCELoss()(y, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print('loss: ', loss.mean())

    test_y = net(x)
    real = torch.argmax(y, dim=1)
    predict = torch.argmax(test_y, dim=1)
    acc = torch.true_divide(torch.sum(predict == real),len(real))
    print('acc: ', acc)

