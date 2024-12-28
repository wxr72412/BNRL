"""
    Defines implementation for NPNs
"""

import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from func.myloss import my_CrossEntropyLoss_novar, my_CE_EM, my_CrossEntropyLoss_KL_MSE, my_MSELoss

PI = float(np.pi)
ETA_SQ = float(np.pi / 8.0)
# for sigmoid
ALPHA = float(4.0 - 2.0 * np.sqrt(2))
ALPHA_SQ = float(ALPHA ** 2.0)
BETA = - float(np.log(np.sqrt(2) + 1))
# for tanh
ALPHA_2 = float(8.0 - 4.0 * np.sqrt(2))
ALPHA_2_SQ = float(ALPHA ** 2.0)
BETA_2 = - float(0.5 * np.log(np.sqrt(2) + 1))

def kappa(x, const=1.0, alphasq= 1.0):
    return 1 / torch.sqrt(const + x * alphasq * ETA_SQ)

loss_func_MSE = nn.MSELoss()

class GaussianNPNKLDivergence(autograd.Function):
    """
        Implements KL Divergence loss for output layer
        KL(N(o_m, o_s) || N(y_m, diag(eps))
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    """
    @staticmethod
    def forward(ctx, o_m, o_s, y, eps):
        ctx.save_for_backward(o_m, o_s, y)
        k = torch.Tensor([y.size(1)])
        det_ratio = torch.prod(o_s / eps, 1)
        KL = (torch.sum(o_s/eps, 1) + torch.sum(torch.pow(o_m - y, 2) / eps, 1) - k + torch.log(det_ratio)) * 0.5
        return torch.Tensor([torch.mean(KL)])
        return torch.mean
        

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


class GaussianNPNLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(GaussianNPNLinearLayer, self).__init__()
        self.W_m, self.M_s = self.get_init_W(input_features, output_features)
        self.b_m, self.p_s = self.get_init_b(output_features)

    def get_init_W(self, _in, _out):
        # obtained from the original paper
        W_m = 2 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out) - 0.5)
        W_s = 1 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out))
        M_s = np.log(np.exp(W_s) - 1)
        return nn.Parameter(torch.FloatTensor(W_m)), nn.Parameter(torch.FloatTensor(M_s))

    def get_init_b(self, _out):
        b_m = np.zeros((_out))
        # instead of bias_s, parametrize as log(1+exp(pias_s))
        p_s = np.exp(-1 * np.ones((_out)))
        return nn.Parameter(torch.Tensor(b_m)), nn.Parameter(torch.Tensor(p_s))

    def forward(self, input):
        if isinstance(input, tuple):
            input_m, input_s = input
        elif isinstance(input, torch.Tensor):
            input_m = input
            input_s = autograd.Variable(input.new().zero_())
        else:
            raise ValueError('input was not a tuple or torch.Tensor (%s)' % type(input))

        # do this to ensure positivity of W_s, b_s
        b_s = torch.log(1 + torch.exp(self.p_s))
        W_s = torch.log(1 + torch.exp(self.M_s))

        o_m = self.b_m + torch.matmul(input_m, self.W_m)
        o_s = b_s + torch.matmul(input_s, W_s) + \
            torch.matmul(input_s, torch.pow(self.W_m, 2)) + \
            torch.matmul(torch.pow(input_m, 2), W_s)
        return o_m, o_s


class GaussianNPNNonLinearity(nn.Module):
    # TODO: does it help to define this in a function instead?
    def __init__(self, activation):
        super(GaussianNPNNonLinearity, self).__init__()
        self.activation = activation

    def forward(self, o):
        o_m, o_s = o
        if self.activation == 'sigmoid':
            a_m = torch.sigmoid(o_m * kappa(o_s))
            a_s = torch.sigmoid(((o_m + BETA) * ALPHA) * kappa(o_s, alphasq=ALPHA_SQ)) - torch.pow(a_m, 2)
            return a_m, a_s
        elif self.activation == 'tanh':
            a_m = 2.0 * torch.sigmoid(o_m * kappa(o_s, const=0.25)) - 1
            a_s = 4.0 * torch.sigmoid(((o_m + BETA_2) * ALPHA_2) * kappa(o_s, alphasq=ALPHA_2_SQ)) - torch.pow(a_m, 2) - 2.0 * a_m - 1.0
            return a_m, a_s
        else:
            return o_m, o_s


class GaussianNPN(nn.Module):
    def __init__(self, input_features, cardinalities, node_type, hidden_sizes, para):
        super(GaussianNPN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        activation = para['activation']
        self.node_type = node_type
        self.layers = []
        self.para = para
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = GaussianNPNLinearLayer(input_features, hidden_sizes[i])
            else:
                h = GaussianNPNLinearLayer(hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(h)
            self.layers.append(GaussianNPNNonLinearity(activation))
        if len(hidden_sizes) > 0:
            self.layers.append(GaussianNPNLinearLayer(hidden_sizes[-1], cardinalities))
        else:
            self.layers.append(GaussianNPNLinearLayer(input_features, cardinalities))
        if self.node_type == 'D':
            self.layers.append(GaussianNPNNonLinearity('sigmoid')) # last one needs to be sigmoid
        # elif self.node_type == 'C':
        #     self.layers.append(GaussianNPNNonLinearity('other')) # last one needs to be sigmoid
        self.net = nn.Sequential(*list(self.layers)) # just to make my_model.parameters() work

        # self.lossfn = nn.BCELoss(reduction='sum')
        # self.lossfn = nn.CrossEntropyLoss(reduction='sum')
        if self.node_type == 'D':
            # self.lossfn = my_CrossEntropyLoss_novar(reduction='sum')
            # self.lossfn = my_CrossEntropyLoss_var(reduction='sum')
            # self.lossfn = my_CrossEntropyLoss_KL_CE(reduction='sum', q_s = 0.01)
            self.lossfn = my_CE_EM(reduction='sum', para=para, q_s=0.01)
        elif self.node_type == 'C':
            # self.lossfn = nn.MSELoss(reduction='sum')
            # self.lossfn = my_MSELoss(reduction='sum')
            self.lossfn = my_CrossEntropyLoss_KL_MSE(reduction='sum', q_s = 0.01)

    def sumnorm(self, output):
        output_m, output_s = output
        # print("output_m:")
        # print(output_m)
        # print(output_m.shape)
        # print()
        # print("output_s:")
        # print(output_s)
        # print(output_s.shape)
        # print()
        output_m_sum = torch.sum(output_m, dim=1).unsqueeze(dim=1)  # 归一化被除数
        # print("output_m_sum:")
        # print(output_m_sum)
        # print(output_m_sum.shape)
        # print()
        output_m_norm = torch.div(output_m, output_m_sum) # 归一化为概率值
        # print("output_m_norm:")
        # print(output_m_norm)
        # print(output_m_norm.shape)  # torch.Size([2, 3])
        # print()
        output_s_norm = torch.div(output_s, torch.pow(output_m_sum, 2))
        # print("output_s_norm:")
        # print(output_s_norm)
        # print(output_s_norm.shape)  # torch.Size([2, 3])
        # print()
        return output_m_norm, output_s_norm

    def forward(self, a):
        # print(a[0][0])
        # print(a[1][0])
        for L in self.layers:
            a = L(a)
        a_m, a_s = a
        # print(a_m)
        # print(a_s)
        # exit(0)
        if self.node_type == 'D':
            a_m, a_s = self.sumnorm(a)
        elif self.node_type == 'C':
            a_m, a_s = a
        # print(a_m)
        # print(a_s)
        # exit(0)
        return a_m, a_s

    def loss(self, x_m, x_s, y, w, prior, print_output = None, kl_target = None):
        a_m, a_s = self.forward((x_m, x_s))
        # print("x_m:")
        # print(x_m)
        # print(x_m.shape)
        # print("x_s:")
        # print(x_s)
        # print(x_s.shape)
        # if print_output == 1:
        # print("a_m:")
        # print(a_m)
        # print(a_m.shape)
        # print("a_s:")
        # print(a_s)
        # print(a_s.shape)
        # print("y:")
        # print(y)
        # print(y.shape)
        # print("weight:")
        # print(weight)
        # print(weight.shape)
        # exit(0)
        if self.node_type == 'D':
            if kl_target == None:
                return self.lossfn((a_m, a_s), y, w, prior, x_m)
            elif kl_target != None:
                # print(kl_target)
                # print(a_m)
                # print(kl_target.div(a_m)) # P/Q
                # print(torch.log(kl_target.div(a_m))) # log(P/Q)
                # print(torch.log(kl_target.div(a_m)).mul(kl_target)) # P*log(P/Q)
                # print(torch.sum(torch.log(kl_target.div(a_m)).mul(kl_target)))  # P*log(P/Q)
                # exit(0)
                BN_loss = self.lossfn((a_m, a_s), y, w, prior, x_m)
                # print(BN_loss)
                kl_loss1 = torch.sum(torch.log(kl_target.div(a_m)).mul(kl_target))
                # kl_loss2 = torch.sum(torch.log(a_m.div(kl_target)).mul(a_m))
                # mse_loss = loss_func_MSE(kl_target.reshape(-1), a_m.reshape(-1))
                # # print(kl_loss)
                # exit(0)
                # return BN_loss
                # return BN_loss + kl_loss1 + mse_loss
                # return BN_loss + kl_loss1 * 1e3
                return BN_loss + kl_loss1 * 5e2
                # return BN_loss + kl_loss1 * 2e2
                # return BN_loss + kl_loss1 * 1e2

                # return BN_loss + mse_loss * 1e3
        elif self.node_type == 'C':
            # return self.lossfn((a_m, a_s), y, w, prior)
            return self.lossfn((a_m, a_s), y)
            # return self.lossfn(a_m, y)




class vanillaNN(nn.Module):
    def __init__(self, input_features, cardinalities, node_type, hidden_sizes, para):
        super(vanillaNN, self).__init__()
        assert(len(hidden_sizes) >= 0)
        activation = para['activation']
        self.node_type = node_type
        self.layers = []
        for i, h_sz in enumerate(hidden_sizes):
            if i == 0 :
                h = nn.Linear(input_features, hidden_sizes[i])
            else:
                h = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            self.layers.append(h)
        if len(hidden_sizes) > 0:
            self.layers.append(nn.Linear(hidden_sizes[-1], cardinalities))
        else:
            self.layers.append(nn.Linear(input_features, cardinalities))
        if self.node_type == 'D':
            self.layers.append(nn.Sigmoid()) # last one needs to be sigmoid
        self.net = nn.Sequential(*list(self.layers)) # just to make my_model.parameters() work

        # self.lossfn = nn.BCELoss(size_average=False)
        # self.lossfn = nn.MSELoss(size_average=False)
        # self.lossfn = nn.CrossEntropyLoss(size_average=False)
        # self.lossfn = my_CrossEntropyLoss_novar(reduction='sum')
        # self.lossfn = my_CrossEntropyLoss(size_average=False)
        # self.lossfn = my_MSELoss(size_average=False)
        if self.node_type == 'D':
            # self.lossfn = my_CrossEntropyLoss_novar(reduction='sum')
            # self.lossfn = my_CrossEntropyLoss_var(reduction='sum')
            # self.lossfn = my_CrossEntropyLoss_KL_CE(reduction='sum', q_s = 0.01)
            self.lossfn = my_CE_EM(reduction='sum', para=para, q_s=0.01)
        elif self.node_type == 'C':
            # self.lossfn = nn.MSELoss(reduction='sum')
            # self.lossfn = my_MSELoss(reduction='sum')
            self.lossfn = my_CrossEntropyLoss_KL_MSE(reduction='sum', q_s = 0.01)

    def sumnorm(self, output):
        output_m = output
        # print("output_m:")
        # print(output_m)
        # print(output_m.shape)
        # print()
        output_m_sum = torch.sum(output_m, dim=1).unsqueeze(dim=1)  # 归一化被除数
        # print("output_m_sum:")
        # print(output_m_sum)
        # print(output_m_sum.shape)
        # print()
        output_m_norm = torch.div(output_m, output_m_sum) # 归一化为概率值
        # print("output_m_norm:")
        # print(output_m_norm)
        # print(output_m_norm.shape)  # torch.Size([2, 3])
        # print()
        return output_m_norm

    def forward(self, a):
        a_m, a_s = a
        for L in self.layers:
            a_m = L(a_m)
            # a_m = torch.sigmoid(a_m)
            a_m = nn.LeakyReLU()(a_m)
        if self.node_type == 'D':
            a_m = self.sumnorm(a_m)
        elif self.node_type == 'C':
            a_m = a_m
        return a_m, None

    # def loss(self, x_m, x_s, y, w, prior, print_output = None):
    #     a_m, a_s = self.forward((x_m, x_s))
    #     # print("a_m:")
    #     # print(a_m)
    #     # print(a_m.shape)
    #     # exit(0)
    #     return self.lossfn(a_m, y)


    def loss(self, x_m, x_s, y, w, prior, print_output = None, kl_target = None):
        a_m, a_s = self.forward((x_m, x_s))
        if self.node_type == 'D':
            if kl_target == None:
                return self.lossfn((a_m, a_s), y, w, prior, x_m)
            elif kl_target != None:
                BN_loss = self.lossfn((a_m, a_s), y, w, prior, x_m)
                kl_loss1 = torch.sum(torch.log(kl_target.div(a_m)).mul(kl_target))
                return BN_loss + kl_loss1 * 5e2
        elif self.node_type == 'C':
            return self.lossfn((a_m, a_s), y)


class rsNNC(nn.Module):
    def __init__(self, para): #  input_features = [z_q[0][input_other[0]], z_q[1][input_other[1]]]
        super(rsNNC, self).__init__()

        user_layers_num = [para['hidden_dim_VQVAE_U'], 32, 32]
        self.user_layers = torch.nn.ModuleList()
        for From, To in zip(user_layers_num[:-1], user_layers_num[1:]):
            self.user_layers.append(nn.Linear(From, To))

        item_layers_num = [para['hidden_dim_VQVAE_I'], 32, 32]
        self.item_layers = torch.nn.ModuleList()
        for From, To in zip(item_layers_num[:-1], item_layers_num[1:]):
            self.item_layers.append(nn.Linear(From, To))

        layers=[64, 64, 32, 8]
        self.finalLayer = torch.nn.Linear(layers[-1] * 2, 1)
        self.gmf_Layer = torch.nn.Linear(32, 8, False)

        self.mlp_layers = torch.nn.ModuleList()
        for From, To in zip(layers[:-1], layers[1:]):
            self.mlp_layers.append(nn.Linear(From, To))

        self.lossfn = nn.MSELoss(size_average=False)

    def forward(self, x):
        uembd = x[0]
        iembd = x[1]
        # print(uembd.shape)
        # print(iembd.shape)
        # exit(0)

        for l in self.user_layers:
            uembd = l(uembd)
            uembd = nn.LeakyReLU()(uembd)

        for l in self.item_layers:
            iembd = l(iembd)
            iembd = nn.LeakyReLU()(iembd)
        # print(uembd.shape)
        # print(iembd.shape)

        MLP_embd = torch.cat([uembd, iembd], dim=1)
        # print(MLP_embd.shape)

        x = MLP_embd
        for l in self.mlp_layers:
            x = l(x)
            x = nn.LeakyReLU()(x)
        # print(x.shape)

        GMF_embd = torch.mul(uembd, iembd)
        y = self.gmf_Layer(GMF_embd)
        y = nn.LeakyReLU()(y)

        NMF_embd = torch.cat([x, y], dim=1)
        prediction = self.finalLayer(NMF_embd)

        return prediction

    def loss(self, x, y):
        prediction = self.forward(x)
        # print(prediction.shape)
        # print(y.shape)
        return self.lossfn(prediction, y)


class rsNND(nn.Module):
    def __init__(self, para): #  input_features = [z_q[0][input_other[0]], z_q[1][input_other[1]]]
        super(rsNND, self).__init__()

        user_layers_num = [para['dim_U_feature'], 32, 32]
        self.user_layers = torch.nn.ModuleList()
        for From, To in zip(user_layers_num[:-1], user_layers_num[1:]):
            self.user_layers.append(nn.Linear(From, To))

        item_layers_num = [para['dim_I_feature'], 32, 32]
        self.item_layers = torch.nn.ModuleList()
        for From, To in zip(item_layers_num[:-1], item_layers_num[1:]):
            self.item_layers.append(nn.Linear(From, To))
        # print(user_layers_num)
        # print(item_layers_num)
        # exit(0)

        layers=[64, 64, 32, 8]
        self.finalLayer = torch.nn.Linear(layers[-1] * 2, 1)
        self.gmf_Layer = torch.nn.Linear(32, 8, False)

        self.mlp_layers = torch.nn.ModuleList()
        for From, To in zip(layers[:-1], layers[1:]):
            self.mlp_layers.append(nn.Linear(From, To))

        self.lossfn = nn.MSELoss(size_average=False)

    def forward(self, x):
        uembd = x[0]
        iembd = x[1]
        # print(uembd.shape)
        # print(iembd.shape)
        # exit(0)

        for l in self.user_layers:
            uembd = l(uembd)
            uembd = nn.LeakyReLU()(uembd)

        for l in self.item_layers:
            iembd = l(iembd)
            iembd = nn.LeakyReLU()(iembd)
        # print(uembd.shape)
        # print(iembd.shape)

        MLP_embd = torch.cat([uembd, iembd], dim=1)
        # print(MLP_embd.shape)

        x = MLP_embd
        for l in self.mlp_layers:
            x = l(x)
            x = nn.LeakyReLU()(x)
        # print(x.shape)

        GMF_embd = torch.mul(uembd, iembd)
        y = self.gmf_Layer(GMF_embd)
        y = nn.LeakyReLU()(y)

        NMF_embd = torch.cat([x, y], dim=1)
        prediction = self.finalLayer(NMF_embd)

        return prediction

    def loss(self, x, y):
        prediction = self.forward(x)
        # print(prediction.shape)
        # print(y.shape)
        return self.lossfn(prediction, y)




