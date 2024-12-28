import torch
import time

def filling_Data(BNML, i, list_latent, data_m, target, para, input_other, VQVAE):
    list_w = []
    list_data_m_EM = []
    list_target_EM = []
    print(data_m[0])
    print(data_m[1])
    print(target[0])
    print(target[1])
    print(i) # 当前节点
    print(list_latent) # 与当前节点相关的隐变量

    sum_w = torch.FloatTensor(data_m.shape[0], 1).zero_().to(para['device'])
    for lv in list_latent:
        ov = BNML.list_correspond_observed_variables[lv] # 与隐变量节点对应的显变量
        # print(lv)
        # print(BNML.list_correspond_observed_variables)
        # print(ov)
        # exit(0)
        for index in range(BNML.list_cardinalities[ov]): # 将对应显变量的取值逐一填充到隐变量取值中
            t1 = time.time()
            temp_data_m = data_m.clone().detach() # 避免修改temp_data_m的时候将data_m修改
            temp_target = target.clone().detach()
            t2 = time.time()
            print("time 1-1 =", "{:.4f}".format(t2 - t1))
            if i == lv: # 隐变量是当前节点，则填充target
                # print(torch.ones(temp_target.shape[0]))
                q = torch.ones(temp_target.shape[0]).to(para['device']) * index
                # print(q.to(int).unsqueeze(1))
                temp_target = torch.FloatTensor(temp_target.shape[0], BNML.list_cardinalities[lv]).zero_().to(para['device']).scatter_(1, q.to(int).unsqueeze(1), 1)
                # print(q)
                # print(temp_target)
                t3 = time.time()
                print("time 1-2 =", "{:.4f}".format(t3 - t2))
                temp_w = get_theta_L_cu(BNML, i, q, ov, temp_data_m, temp_target, para, input_other, VQVAE)
                t4 = time.time()
                print("time 1-3 =", "{:.4f}".format(t4 - t3))
            else: # 隐变量是当前节点的父节点，
                q = torch.ones(temp_data_m.shape[0]).to(para['device']) * index # 如果有2条数据，当前填充取值index=1，则tensor([1., 1.])
                xx = torch.FloatTensor(temp_data_m.shape[0], BNML.list_cardinalities[lv]).zero_().to(para['device']).scatter_(1, q.to(int).unsqueeze(1), 1) # 将tensor([1., 1.])变为one-hot形式
                temp_data_m[:, sum(BNML.list_cardinalities[:lv]):sum(BNML.list_cardinalities[:lv+1])] = xx # 填充隐变量对应的取值
                temp_w = get_theta_L_pa(BNML, i, q, ov, temp_data_m, temp_target, para)
            sum_w += temp_w
            list_data_m_EM.append(temp_data_m)
            list_target_EM.append(temp_target)
            list_w.append(temp_w)
            t5 = time.time()
            print("time EM 1-4 =", "{:.4f}".format(t5 - t4))

    t1 = time.time()
    data_m_EM = torch.cat([l for l in list_data_m_EM], 0) # 将隐变量不同取值合并在一起
    t2 = time.time()
    print("time 3-1 =", "{:.4f}".format(t2 - t1))
    target_EM = torch.cat([l for l in list_target_EM], 0)
    t3 = time.time()
    print("time 3-2 =", "{:.4f}".format(t3 - t2))
    w = torch.cat([l for l in list_w], 0)
    t4 = time.time()
    print("time 3-3 =", "{:.4f}".format(t4 - t3))
    w = (w.reshape(-1, data_m.shape[0]) / sum_w.reshape(-1)).reshape(data_m_EM.shape[0],-1)
    t5 = time.time()
    print("time 3-4 =", "{:.4f}".format(t5 - t4))
    # print(data_m_EM[1])
    # print(data_s_EM)
    # print(target_EM[1])
    # print(w)

    # print(data_m_EM)
    # print(target_EM)
    # print(w)
    #
    # print(data_m_EM.shape)
    # print(target_EM.shape)
    # print(w.shape)
    # exit(0)

    return data_m_EM, target_EM, w


def get_theta_L_cu(BNML, i, q, ov, temp_data_m, temp_target, para, input_other, VQVAE):
    if para['data_file'] == 'ml-1m': # L是当前节点
        I_index = ov
        L_index = i
        R_index = L_index + para['dim_I_feature']
        # print(I_index) # 12
        # print(L_index) # 24
        # print(R_index) # 36
        # exit(0)

        if BNML.list_node_label[I_index] == 'eDIA':
            index = 0 # 取出第index列的item嵌入特征
            for k in range(BNML.num_nodes):
                if BNML.list_node_label[k] == 'eDIA':
                    if k == I_index:
                        break
                    if k != I_index:
                        index += 1
        # print(index)  # 0

        j = VQVAE.indexI[:, index][input_other[1]]
        # print(j) # tensor([1, 9], device='cuda:0')
        k = q.reshape(-1).to(int)
        # print(k) # tensor([0, 0], device='cuda:0')
        index = BNML.list_cardinalities[I_index] * j + k
        # print(index) # tensor([10, 90], device='cuda:0')
        temp_theta = BNML.list_init_theta[R_index][index]
        # print(temp_theta)
        # tensor([[0.2300, 0.2200, 0.2100, 0.1800, 0.1600],
        #         [0.2300, 0.2200, 0.2100, 0.1800, 0.1600]], device='cuda:0')
        r = (input_other[2]-1).reshape(-1, 1)
        # print(r)
        # tensor([[4],
        #         [2]], device='cuda:0')
        w = temp_theta.gather(1, r)
        # print(w)
        # tensor([[0.1600],
        #         [0.2100]], device='cuda:0')
        # exit(0)
    return w


def get_theta_L_pa(BNML, i, q, ov, temp_data_m, temp_target, para):
    if para['data_file'] == 'ml-1m': # L是当前节点R的父节点
        I_index = ov
        R_index = i

        j = ((temp_data_m[:, sum(BNML.list_cardinalities[:ov]):sum(BNML.list_cardinalities[:ov+1])]).argmax(dim=1)).reshape(-1)
        # print(j) # tensor([1, 9], device='cuda:0')
        # print(temp_data_m) #
        # print(q)
        k = q.reshape(-1).to(int)
        # print(k) # tensor([0, 0], device='cuda:0')
        index = BNML.list_cardinalities[I_index] * j + k
        # print(index) # tensor([10, 90], device='cuda:0')
        temp_theta = BNML.list_init_theta[R_index][index]
        # print(temp_w)
        # tensor([[0.2300, 0.2200, 0.2100, 0.1800, 0.1600],
        #         [0.2300, 0.2200, 0.2100, 0.1800, 0.1600]], device='cuda:0')
        r = (temp_target.argmax(dim=1)).reshape(-1, 1)
        # print(r)
        # tensor([[4],
        #         [2]], device='cuda:0')
        w = temp_theta.gather(1, r)
        # print(w)
        # exit(0)
        # tensor([[0.1600],
        #         [0.2100]], device='cuda:0')
    return w