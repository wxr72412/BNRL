from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss, _WeightedLoss

from torch.overrides import (has_torch_function, has_torch_function_unary, has_torch_function_variadic, handle_torch_function)




Tensor = torch.Tensor





class my_CrossEntropyLoss_novar(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(my_CrossEntropyLoss_novar, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_novar(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

def my_cross_entropy_novar(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):

    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            my_cross_entropy_novar,
            (input, target),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input
    # print(F.nll_loss(torch.log(a_m), target, weight, None, ignore_index, None, reduction))
    # exit(0)

    # 离散变量--交叉熵（行，但方差过大）
    return F.nll_loss(torch.log(a_m), target, weight, None, ignore_index, None, reduction)










class my_CE_EM(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'para']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', para: dict = {}, q_s: float = 0.01) -> None:
        super(my_CE_EM, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.q_s = q_s
        self.para = para

    def forward(self, input: Tensor, target: Tensor, w: Tensor, prior: Tensor, Se: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_EM(input, target, w, prior, Se, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction, para=self.para, q_s=self.q_s)

def my_cross_entropy_EM(
    input: Tensor,
    target: Tensor,
    w: Tensor,
    prior: Tensor,
    Se: Tensor,
    q_s: float,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    para: dict = {},
):

    if has_torch_function_variadic(input, target, q_s, para):
        return handle_torch_function(
            my_cross_entropy_EM,
            (input, target, w, prior, Se, q_s, para),
            input,
            target,
            w,
            prior,
            Se,
            q_s,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            para=para,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input
    # print(a_m)
    # exit(0)
    # CE = F.nll_loss(torch.log(a_m), target, w, None, ignore_index, None, reduction)
    # return CE

    log_a_m = torch.log(a_m)
    # if para['Train'] == 'False':
    #     log_a_m = torch.log(target)
    # print('loss a_m:')
    # print(a_m)
    # print('loss log_a_m:')
    # print(log_a_m)
    # exit(0)
    target_onehot = target
    # if para['Train'] == 'False':
    #     target_onehot = a_m
    # print('target_onehot:')
    # print(target_onehot)
    # exit(0)
    if prior != None:
        # print('prior:')
        # print(prior)
        target_onehot += prior
    # print('target_onehot:')
    # print(target_onehot)
    # exit(0)
    negative_log_a_m_onehot = log_a_m.mul(target_onehot) * (-1)
    # print('loss negative_log_a_m_onehot:')
    # print(negative_log_a_m_onehot)
    # print('w:')
    # print(w)
    N = 1
    if reduction == "mean":
       N = target.shape[0]
    # print(reduction)

    if w == None:
        pass
    else:
        # print("w:")
        # print(w)
        # print(w.shape)
        negative_log_a_m_onehot = negative_log_a_m_onehot.mul(w)
    # print('negative_log_a_m_onehot:')
    # print(negative_log_a_m_onehot)
    loss = negative_log_a_m_onehot.sum() / N

    # print()
    # print(loss)
    # exit(0)

    # if para['MAP'] == 'Likelihood+gt0':
    #     # loss = loss + (1-torch.exp(a_m)).sum()
    #     loss = loss + (-torch.log(a_m)).sum()
    #     # print(loss)
    # if para['MAP'] == 'Likelihood+gt0+sum1':
    #     # loss = loss + abs(a_m).sum()
    #     # print(a_m)
    #     # print(torch.sum(a_m, 1) - 1) # 按行求和
    #     # print(a_m.shape[0])
    #     # loss = loss + (-torch.log(a_m)).sum() + torch.pow(torch.sum(a_m, 1) - 1, 2).sum()
    #     print('a_m:')
    #     print(a_m)
    #     print('torch.log(a_m):')
    #     print(torch.log(a_m))
    #     print('torch.log(1-a_m):')
    #     print(torch.log(1-a_m))
    #     print('Se:')
    #     print(Se)
    #     # loss = loss + (-torch.log(Se)).sum()
    #     # loss = loss + (-torch.log(1-Se)).sum()
    #     loss = loss + (-torch.log(Se)).sum() + (-torch.log(1-Se)).sum()
    #     print(loss)
    #     # exit(0)

    return loss







class my_CrossEntropyLoss_var(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(my_CrossEntropyLoss_var, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_var(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

def my_cross_entropy_var(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):

    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            my_cross_entropy_var,
            (input, target),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input
    # print(a_m)
    log_a_m = torch.log(a_m)
    # print(log_a_m)
    # print(a_s)

    target_onehot = torch.FloatTensor(target.shape[0], 2)
    target_onehot.zero_()
    # print(target)
    # print(target.shape)
    negative_log_a_m_onehot = log_a_m.mul(target_onehot.scatter_(1, target.unsqueeze(1), 1)) * (-1)
    # print(negative_log_a_m_onehot)
    # exit(0)

    # 连续变量--对数似然，并将误差平方替换为负对数似然（不行，方差过小导致nan）
    return (negative_log_a_m_onehot.div(a_s) + torch.log(a_s)).sum()
    # return (negative_log_a_m_onehot + torch.abs(a_s)).sum()

    # 离散变量--交叉熵，等价于my_CrossEntropyLoss_novar（行，但方差过大）
    # return negative_log_a_m_onehot.sum()








# class my_CrossEntropyLoss_var_onehot(_WeightedLoss):
#     __constants__ = ['ignore_index', 'reduction']
#     ignore_index: int
#
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
#                  reduce=None, reduction: str = 'mean') -> None:
#         super(my_CrossEntropyLoss_var_onehot, self).__init__(weight, size_average, reduce, reduction)
#         self.ignore_index = ignore_index
#
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         assert self.weight is None or isinstance(self.weight, Tensor)
#         return my_cross_entropy_novar_onehot(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)
#
# def my_cross_entropy_novar_onehot(
#     input: Tensor,
#     target: Tensor,
#     weight: Optional[Tensor] = None,
#     size_average: Optional[bool] = None,
#     ignore_index: int = -100,
#     reduce: Optional[bool] = None,
#     reduction: str = "mean",
# ):
#
#     if has_torch_function_variadic(input, target):
#         return handle_torch_function(
#             my_cross_entropy_novar_onehot,
#             (input, target),
#             input,
#             target,
#             weight=weight,
#             size_average=size_average,
#             ignore_index=ignore_index,
#             reduce=reduce,
#             reduction=reduction,
#         )
#     if size_average is not None or reduce is not None:
#         reduction = _Reduction.legacy_get_string(size_average, reduce)
#     a_m, a_s = input
#     log_a_m = torch.log(a_m)
#     negative_log_a_m_onehot = log_a_m.mul(target.scatter_(1, target.unsqueeze(1), 1)) * (-1)
#     return negative_log_a_m_onehot.sum()



class my_CrossEntropyLoss_KL_CE(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', q_s: float = 0.01) -> None:
        super(my_CrossEntropyLoss_KL_CE, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.q_s = q_s

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_KL_CE(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction, q_s = self.q_s)

def my_cross_entropy_KL_CE(
    input: Tensor,
    target: Tensor,
    q_s: float,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "sum",
):

    if has_torch_function_variadic(input, target, q_s):
        return handle_torch_function(
            my_cross_entropy_KL_CE,
            (input, target, q_s),
            input,
            target,
            q_s,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input

    # KL(N(μ,σ^2)||N(p,ε^2))（前向KL） 行
    # print(a_m)
    # print(a_s)
    # print(q_s)
    CE = F.nll_loss(torch.log(a_m), target, weight, None, ignore_index, None, reduction)
    # print(CE)
    a_s_div_q_s = a_s.div(q_s)
    # print(a_s_div_q_s)
    # print((torch.log(a_s_div_q_s) * (-1) + a_s_div_q_s).sum() + CE.div(q_s))
    # exit(0)
    return (torch.log(a_s_div_q_s) * (-1) + a_s_div_q_s).sum() + CE.div(q_s)



    # # 离散变量--交叉熵CE+KL(N(μ,σ^2)||N(μ,ε^2))（前向KL） 不行（输出全是均值）
    # # print(a_m)
    # # print(a_s)
    # # print(q_s)
    # CE = F.nll_loss(torch.log(a_m), target, weight, None, ignore_index, None, reduction)
    # # print(CE)
    # a_s_div_q_s = a_s.div(q_s)
    # # print(a_s_div_q_s)
    # # exit(0)
    # return (torch.log(a_s_div_q_s) * (-1) + a_s_div_q_s).sum() + CE





    # # KL(N(p,ε^2)||N(μ,σ^2)) （后向KL） 行
    # log_a_m = torch.log(a_m)
    # target_onehot = torch.FloatTensor(target.shape[0], 2)
    # target_onehot.zero_()
    # negative_log_a_m_onehot = log_a_m.mul(target_onehot.scatter_(1, target.unsqueeze(1), 1)) * (-1)
    # q_s_div_a_s = torch.pow(a_s, -1) * q_s
    # print(q_s_div_a_s)
    # # print(negative_log_a_m_onehot)
    # # exit(0)
    # return ((torch.log(q_s_div_a_s) * (-1) + q_s_div_a_s) + negative_log_a_m_onehot.div(a_s)).sum()








class my_CrossEntropyLoss_KL_MSE(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', q_s: float = 0.01) -> None:
        super(my_CrossEntropyLoss_KL_MSE, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.q_s = q_s

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_KL_MSE(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction, q_s=self.q_s)

def my_cross_entropy_KL_MSE(
    input: Tensor,
    target: Tensor,
    q_s: float,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):

    if has_torch_function_variadic(input, target, q_s):
        return handle_torch_function(
            my_cross_entropy_KL_MSE,
            (input, target, q_s),
            input,
            target,
            q_s,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input

    # # KL(N(μ,σ^2)||N(p,ε^2))（前向KL） 行
    # a_s_div_q_s = a_s.div(q_s)
    # return (torch.log(a_s_div_q_s) * (-1) + a_s_div_q_s + torch.pow(a_m - target, 2).div(q_s)).sum()

    # # KL(N(p,ε^2)||N(μ,σ^2)) （后向KL） 行
    # q_s_div_a_s = torch.pow(a_s, -1) * q_s
    # return (torch.log(q_s_div_a_s) * (-1) + q_s_div_a_s + torch.pow(a_m - target, 2).div(a_s)).sum()


    # print(target)
    # print(a_m)
    # 交叉熵 P*log(1/Q) = -P*log/Q
    # print(torch.log(a_m))
    # print(torch.log(a_m).mul(target))
    # print(torch.log(a_m).mul(target) * -1)
    # loss = torch.log(a_m).mul(target) * -1
    # exit(0)

    # 均方误差
    # print('a_m - target')
    # print(a_m - target)
    loss = torch.pow(a_m - target, 2)
    # print(loss)
    # exit(0)

    # KL散度 # P*log(P/Q)
    # print(target.div(a_m)) # P/Q
    # print(torch.log(target.div(a_m))) # log(P/Q)
    # print(torch.log(target.div(a_m)).mul(target)) # P*log(P/Q)
    # print(torch.sum(torch.log(target.div(a_m)).mul(target)))  # P*log(P/Q)
    # loss = torch.log(target.div(a_m)).mul(target)
    # KL散度 # Q*log(Q/P)
    # loss = torch.log(a_m.div(target)).mul(a_m)

    N = 1
    if reduction == "mean":
        N = target.shape[0]
    loss = loss.sum() / N

    return loss











# class my_MSELoss(_Loss):
#     __constants__ = ['reduction']
#
#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(my_MSELoss, self).__init__(size_average, reduce, reduction)
#
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         a_m, a_s = input
#         # print('a_m')
#         # print(a_m)
#         # print(a_s)
#         # print('target')
#         # print(target)
#         # print(a_m - target)
#         # print(torch.pow(a_m - target, 2))
#         # print((torch.pow(a_m - target, 2).div(a_s) + torch.log(a_s)).sum().item())
#         # exit(0)
#         # print((torch.pow(a_m - target, 2).div(a_s) + torch.log(a_s)).div(2))
#         return (torch.pow(a_m - target, 2).div(a_s) + torch.log(a_s)).sum()
#         # return (torch.pow(a_m - target, 2)).sum()



class my_MSELoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'para']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(my_MSELoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, w: Tensor, prior: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_MSELoss_EM(input, target, w, prior,
                                   ignore_index=self.ignore_index, reduction=self.reduction)

def my_MSELoss_EM(
        input: Tensor,
        target: Tensor,
        w: Tensor,
        prior: Tensor,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
):

    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            my_cross_entropy_EM,
            (input, target, w, prior),
            input,
            target,
            w,
            prior,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input

    loss = (torch.pow(a_m - target, 2).div(a_s) + torch.log(a_s))

    # print(a_m)
    # print(target)

    if prior != None:
        numerator = torch.pow(a_m - prior, 2)
        # loss += (numerator.div(a_s))
        loss += (numerator.div(a_s) + torch.log(a_s))
        # numerator = torch.pow(a_m - prior, 2)
        # loss += numerator
    # exit(0)

    if w == None:
        None
    else:
        loss = loss.mul(w)

    N = 1
    if reduction == "mean":
        N = target.shape[0]
    loss = loss.sum() / N

    return loss