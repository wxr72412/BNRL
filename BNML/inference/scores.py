# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import heapq
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def topK_scores(test_user_pre_label, predcit_user_pre, topk, user_num):
    PrecisionSum = np.zeros(topk+1)
    # print(PrecisionSum)
    RecallSum = np.zeros(topk+1)
    # print(RecallSum)
    F1Sum = np.zeros(topk+1)
    # print(F1Sum)
    NDCGSum = np.zeros(topk+1)
    # print(NDCGSum)
    # OneCallSum = np.zeros(topk+1)
    # print(OneCallSum)
    DCGbest = np.zeros(topk+1)
    auc_scoreSum = 0
    # print(DCGbest)
    # exit(0)

    # MRRSum = 0
    MAPSum = 0


    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    # print(DCGbest)
    # exit(0)



    for i in range(0, user_num):
        # print("user:" + str(i))
        user_predict = predcit_user_pre[i]
        user_test = test_user_pre_label[i]
        # print(user_test)
        # print(user_predict)
        test_data_size = int(sum(user_test))
        # exit(0)


        # AUC
        auc_scoreSum += roc_auc_score(user_test, user_predict)

        predict_max_num_index_list = map(user_predict.index, heapq.nlargest(topk, user_predict))  # 在集合中找出最大的topk个元素对应的index  # heapq.nlargest(topk, user_predict) # 在集合中找出最大的topk个元素
        # print(predict_max_num_index_list)
        predict_max_num_index_list = list(predict_max_num_index_list)
        # exit(0)

        # MAP
        p = 1
        AP = 0.0
        hit_before = 0
        # print(predict_max_num_index_list)
        # print(user_test)
        for mrr_iter in predict_max_num_index_list:
            # print('p: ' + str(p) + ' hb: ' + str(hit_before))
            if user_test[mrr_iter] == 1: # 当前预测正确
                AP += 1 / float(p) * (hit_before + 1)
                hit_before += 1
            p += 1
        #     print('AP: ' + str(AP))
        # print(test_data_size)
        MAPSum += AP / test_data_size
        # print('MAPSum: ' + str(MAPSum) + ' AP: ' + str(AP) + ' test_data_size: ' + str(test_data_size))
        # exit(0)




        # print(user_test)  # [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(user_predict)  # [0.31, 0.5, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # print(test_data_size)  # 3
        # print(predict_max_num_index_list) # [1, 2, 0]
        # exit(0)

        hit_sum = 0
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)

        for k in range(1, topk + 1): # k = 1,2, ..., topk
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1] #
            if user_test[item_id] == 1:
                hit_sum += 1
                DCG[k] += 1 / math.log(k + 1)

            # precision, recall, F1, 1-call
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            f1 = 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            F1Sum[k] += float(f1)
            # print("PRF")
            # print(PrecisionSum)
            # print(RecallSum)
            # print(F1Sum)
            # print()

            # NDCG
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]

            # if hit_sum > 0:
            #     OneCallSum[k] += 1
            # else:
            #     OneCallSum[k] += 0

        # print("PRF")
        # print(PrecisionSum)
        # print(RecallSum)
        # print(F1Sum)
        # print()

        # MRR
        # p = 1
        # for mrr_iter in predict_max_num_index_list:
        #     if user_test[mrr_iter] == 1:
        #         break
        #     p += 1
        # MRRSum += 1 / float(p)

    # print(auc_scoreSum)
    # print(PrecisionSum)
    # print(RecallSum)
    # print(F1Sum)
    # print(NDCGSum)

    total_test_data_count = user_num
    # print(total_test_data_count)
    # exit(0)



    print('AUC:', auc_scoreSum / total_test_data_count)
    # AUC、MAP、Rec@n、NDCG@n
    print('MAP:', MAPSum / total_test_data_count)
    # print('MRR:', MRRSum / total_test_data_count)
    print('Prec@:' + str(topk), PrecisionSum[topk] / total_test_data_count)
    print('Rec@:' + str(topk), RecallSum[topk] / total_test_data_count)
    print('F1@:' + str(topk), F1Sum[topk] / total_test_data_count)
    print('NDCG@:' + str(topk), NDCGSum[topk] / total_test_data_count)
    # print('1-call@5:', OneCallSum[topk] / total_test_data_count)

    # exit(0)

    return
