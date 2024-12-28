from learning.AE_leraning_process import AE_learning
import numpy as np
import torch.optim as optim


# def AE_train(AE, AE_type, para, features, adj, norm, weight_A, BNML = None, i = None,
#                               data_m = None, data_s = None, target = None, w = None, prior = None, input_other = None, batch_idx = 0):
#
#     optimizer = optim.Adam(AE.parameters(), lr = para['lr']) # lr
#     best_iter = -1
#     best_loss = np.inf
#     convergence_iter_num = 0
#
#     for epoch in range(1, para['parameter_learning_max_iter_num'] + 1):
#         train_loss, Z = AE_learning(AE, AE_type, optimizer, epoch, para, features, adj, norm, weight_A, BNML, i, data_m, data_s, target, w, prior, input_other)
#         # early stopping
#         if train_loss < best_loss:
#             best_loss = train_loss
#             best_iter = epoch
#             convergence_iter_num = 0
#         else:
#             convergence_iter_num += 1
#         if convergence_iter_num == para['max_convergence_iter_num'] or epoch == para['parameter_learning_max_iter_num']:
#             print('batch_idx: {}, Train Epoch: {}, best_loss: {:.6f} best_iter: {} convergence_iter_num: {}'.format(batch_idx, epoch, best_loss, best_iter, convergence_iter_num))
#             return train_loss, Z

def AE_train1(VQVAE, VQVAE_type, para, features):
    params_list = []
    if VQVAE_type == True:
        params_list.append(VQVAE.parameters())
        # optimizer_VGAE = optim.Adam(VQVAE.parameters(), lr = para['lr']) # lr
    optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr']) # lr
    # print(optimizer)
    # exit(0)

    best_iter = -1
    best_loss = np.inf
    convergence_iter_num = 0
    for epoch in range(1, para['AE_parameter_learning_max_iter_num'] + 1):
        train_loss, Z = AE_learning(VQVAE, VQVAE_type, optimizer, epoch, para, features)
        # early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else:
            convergence_iter_num += 1
        if convergence_iter_num == para['AE_max_convergence_iter_num'] or epoch == para['AE_parameter_learning_max_iter_num']:
            print('Train Epoch: {}, best_loss: {:.6f} best_iter: {} convergence_iter_num: {}'.format(epoch, best_loss, best_iter, convergence_iter_num))
            return train_loss, Z