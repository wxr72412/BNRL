from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import random
# ACC 准确度
#（TP+TN）/N
# 识别正确的数量占总数量的比例。因为实际中一般不会用到TN所以这个一般不用。


# AUC(ROC曲线下的面积):正例排在负例前面的概率。
# AUC = 1，是完美分类器，分类100% 正确。
# 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
# AUC = 0.5，跟随机猜测一样，模型没有预测价值。
# AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

# AP(PR曲线下的面积):单个类别的检测精度
# mAP:对所有类别的AP取平均值

def get_scores(adj_rec, adj_label):


    labels_all = adj_label.to_dense().view(-1)

    # preds_all = (adj_rec > 0.9).view(-1).float()
    preds_all = (adj_rec > 0.6).view(-1).float()
    # preds_all = (adj_rec > 0.5).view(-1).float()

    labels_edge = (labels_all >= 1).long()
    labels_noedge = (labels_all == 0).long()
    preds_edge = (preds_all >= 1).long()
    preds_noedge = (preds_all == 0).long()

    # print(labels_all.size(0))
    # print(labels_edge)
    # print(labels_edge.sum())
    # print(labels_noedge)
    # print(labels_noedge.sum())
    # print(labels_edge+labels_noedge)
    # print((labels_edge+labels_noedge).sum())
    # print((preds_all == labels_edge).sum())
    # print((preds_edge * labels_edge).sum())
    # print((preds_noedge * labels_noedge).sum())
    # print(adj_rec)
    # print(adj_rec.shape)
    # exit(0)

    accuracy = (preds_all == labels_edge).sum().float() / labels_all.size(0)
    edge_accuracy = (preds_edge * labels_edge).sum().float() / labels_edge.sum().float()
    noedge_accuracy = (preds_noedge * labels_noedge).sum().float() / labels_noedge.sum().float()
    # print(accuracy)
    # print(edge_accuracy)
    # print(noedge_accuracy)
    # # exit(0)


    preds_all_1 = adj_rec.detach().view(-1)
    preds_pos = preds_all_1 * labels_edge
    preds_pos = preds_pos[(preds_pos>0)]
    # print(len(preds_pos))

    preds_neg = preds_all_1 * labels_noedge
    # preds_neg = preds_neg[(preds_neg>0)]
    index = random.randint(0, len(preds_neg)-len(preds_pos)) # random.randint(1, 2) # 生成1或2
    # print(index)
    preds_neg = preds_neg[(preds_neg>0)][index:index+len(preds_pos)]
    # print(len(preds_neg))

    # preds = preds_all_1[(labels_all >= 1)]
    # preds_neg = preds_all_1[(labels_all == 0)]

    preds_all_2 = torch.hstack([preds_pos, preds_neg]).cpu()
    # print(preds_all_2)
    # print(preds_all_2.shape)
    # labels_all = torch.hstack([torch.ones(len(preds_pos)), torch.zeros(len(preds_neg))])
    labels_all = torch.hstack([torch.ones(len(preds_pos)), torch.zeros(len(preds_neg))])
    # print(labels_all)
    # print(labels_all.shape)
    # exit(0)
    # print(labels_all.sum())



    roc_score = roc_auc_score(labels_all, preds_all_2)
    ap_score = average_precision_score(labels_all, preds_all_2)
    # print(roc_score)
    # print(ap_score)
    # exit(0)


    return accuracy, edge_accuracy, noedge_accuracy, roc_score, ap_score
    # return accuracy
