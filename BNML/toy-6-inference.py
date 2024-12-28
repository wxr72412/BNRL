from inference.scores import topK_scores
import os
import para_init
import heapq
import copy

def read_file_pre(test_data_dir):
    # print(test_data_dir)
    test_user_pre = [0 for i in range(para_init.num_infer)]
    # print(test_user_pre)
    # exit(0)

    test_file = open(test_data_dir, 'r', encoding='utf-8')
    # print(test_file.readlines())
    # exit(0)

    for i in range(para_init.num_infer):
        # num += 1
        # print("第 " + str(num)+ " 条测试集中的用户评分！")
        line_str = test_file.readline()[0:-1] # 消去末尾的换行符
        # print(line_str)
        # exit(0)
        line_str_split = line_str.split(",")
        # print(line_str_split)
        line_str_split = [float(x) for x in line_str_split]
        user_id = int(line_str_split[0])
        # print(user_id)
        # print(line_str_split)
        # print(line_str_split[1:])
        # exit(0)
        test_user_pre[user_id-1] = line_str_split[1:]
    test_file.close()
    # print(test_user_pre[0])
    # print(test_user_pre[1])
    # for i in test_user_pre:
    #     print(i)
    # exit(0)
    return test_user_pre





def pre(model_type, real_pre_num):
    path_inference = os.path.dirname(__file__) + '\\' + 'inference\\'


    # print("1. 读取VE的结果")
    test_data_dir = path_inference + para_init.bn + "_" + 'VE' + ".txt"
    # print(test_data_dir)
    test_result = read_file_pre(test_data_dir)
    # print(test_result[0])
    # for i in test_result:
    #     print(i)
    # exit(0)


    # print("2. 读取模型的用户偏好")
    train_data_dir = path_inference + para_init.bn + "_" + model_type + ".txt"
    # print(train_data_dir)
    train_result = read_file_pre(train_data_dir)
    # print(train_result[0])
    # for i in train_result:
    #     print(i)
    # exit(0)

    # print("3. 取概率最大的几个值并置1")
    test_result_label = copy.deepcopy(test_result)
    train_result_label = copy.deepcopy(train_result)

    for i in range(0, para_init.num_infer):
        temp_test_result = test_result_label[i]
        # print(temp_test_result)
        predict_max_num_index_list = list(map(temp_test_result.index, heapq.nlargest(real_pre_num, temp_test_result)))
        # print(predict_max_num_index_list)
        for j in range(len(temp_test_result)):
            if j in predict_max_num_index_list:
                test_result_label[i][j] = 1
            else:
                test_result_label[i][j] = 0
        # print(test_result_label[i])
        # exit(0)

    for i in range(0, para_init.num_infer):
        temp_train_result = train_result_label[i]
        # print(temp_train_result)
        predict_max_num_index_list = list(map(temp_train_result.index, heapq.nlargest(real_pre_num, temp_train_result)))
        # print(predict_max_num_index_list)
        for j in range(len(temp_train_result)):
            if j in predict_max_num_index_list:
                train_result_label[i][j] = 1
            else:
                train_result_label[i][j] = 0

    # for i in range(0, para_init.num_infer):
    #     print(test_result_label[i])
    #     print(test_result[i])
    #     print(train_result_label[i])
    #     print(train_result[i])
    #     print()
    # exit(0)

    for top_k in top_k_list:
        # if top_k != real_pre_num:
        #     continue
        print("计算 top_k： " + str(top_k))
        # Top-K evaluation
        str(topK_scores(test_result_label, train_result, top_k, para_init.num_infer))
        print()
        # exit(0)






import time
start_time = time.time()

# list_model_type = 'VE'
list_model_type = ['VE', 'FS', "RS", "GS", "BNML"]
# list_model_type = ['VE', 'FS', "RS", "BNML"]
# list_model_type = ['FS']
# list_model_type = ["RS"]
# list_model_type = ["GS"]
# list_model_type = ["BNML"]


real_pre_num_list = [2]
# real_pre_num_list = [8]
top_k_list = [2]
# top_k_list = [8]

for real_pre_num in real_pre_num_list:
    for model_type in list_model_type:
        print(model_type)
        pre(model_type, real_pre_num)

# real_pre_num_list = [2]
# top_k_list = [2]
# for real_pre_num in real_pre_num_list:
#     for model_type in list_model_type:
#         print(model_type)
#         pre(model_type, real_pre_num)

print("time=", "{:.2f}".format(time.time() - start_time))