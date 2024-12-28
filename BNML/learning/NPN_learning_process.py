import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F
from func.EM import filling_Data

loss_func_MSE = nn.MSELoss()


def NPN_learning(BNML, i, model, optimizer, epoch, para, data_m, data_s, target):
    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # 对于BN是保证BN层能够用到每一批数据的均值和方差。对于Dropout，my_model.train()是随机取一部分网络连接来训练更新参数。
    model.train()
    train_loss = 0
    print_output = 0
    # if epoch % para['print_interval'] == 0:
    #     print_output = 1

    w = None
    prior = None

    optimizer.zero_grad()  # 清空过往梯度
    loss = model.loss(data_m, data_s, target, w, prior, print_output)
    loss.backward() # 反向传播，计算当前梯度
    optimizer.step() # 根据梯度更新网络参数
    train_loss += loss.item()

    # if 1:
    if epoch % para['print_interval'] == 0:

        print('Node: {}, Train Epoch: {}, Train loss: {:.4f}, Batch size: {}\n'.format
              (i+1, epoch, train_loss , len(data_m)))
        output = model((data_m, data_s))
        # print(output[0])
        # print(output[1])

    return train_loss, train_loss