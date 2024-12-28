import numpy as np
import pandas as pd


# %%
# MSE = 1/m * sum(y-t)^2
def MSE(cur_cpt, ori_cpt):
    return np.sum((cur_cpt - ori_cpt) ** 2) / len(cur_cpt)

def MAE(cur_cpt, ori_cpt):
    # print(cur_cpt)
    # print(ori_cpt)
    # print(cur_cpt - ori_cpt)
    # print(len(cur_cpt))
    return np.sum(np.abs(cur_cpt - ori_cpt)) / len(cur_cpt)


# t = np.array([0, 1, 0, 0])
# y = np.array([0.1, 0.05, 0.05, 0.8])
# print(MSE(y, t))
# ori_cpts = [np.array([0.1, 0.4]), np.array[0.3, 0.4]]
# cur_cpts = [np.array([0.2, 0.3]), np.array[0.5, 0.6]]
# for i in cur_cpts:
#     for j in ori_cpts:
# print(MSE(cur_cpts, ori_cpts))


# %%
# KL = sum(p * log(p/q))
def KL(ori_cpt, cur_cpt, car):
    # print(ori_cpt)
    # print(cur_cpt)
    # print(car)
    sum = 0
    div_num = len(ori_cpt)
    for ori_iter, cur_iter in zip(ori_cpt, cur_cpt):
        # if ori_iter != 0 and cur_iter != 0 and not np.isnan(cur_iter):
        if ori_iter == 0 and cur_iter == 0:
            # print('ori_iter:', ori_iter, 'cur_iter:', cur_iter)
            sum += 0
        elif ori_iter == 0 and cur_iter != 0:
            # print('ori_iter:', ori_iter, 'cur_iter:', cur_iter)
            sum += 0

        elif ori_iter != 0 and cur_iter == 0:
            cur_iter = 0.001
            sum += ori_iter * np.log(ori_iter / cur_iter)
        elif ori_iter != 0 and cur_iter == 1:
            cur_iter = 1 - 0.01 * car
            sum += ori_iter * np.log(ori_iter / cur_iter)

        else:
            sum += ori_iter * np.log(ori_iter / cur_iter)
            # print('ori_iter:', ori_iter, 'cur_iter:', cur_iter, 'np.log(ori_iter / cur_iter):', ori_iter * np.log(ori_iter / cur_iter), 'sum:', sum)
    return sum/len(ori_cpt)
    # return sum/div_num


# t = [0, 1, 0, 0]
# y = [0.1, 0.05, 0.05, 0.8]
# print(KL(t, y))
