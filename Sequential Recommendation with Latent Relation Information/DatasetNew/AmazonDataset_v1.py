import torch
from torch.utils.data import Dataset, DataLoader
from util.myutil import *
from scipy.sparse import csr_matrix

import numpy as np

'''
关于数据泄漏问题: 采用短序列必然会造成部分样本在训练的时候就行发生泄漏，即目标序列出现在输入序列中
对训练集和验证集建立图是否合理？
图的建立方式该如何考虑？
1）从用户序列按照时间先后顺序进行建图
2）从用户物品二分图中将两个物品经过1个用户相连则也存在边
模型中数据泄露问题该如何处理？
 也可以不考虑，因为我们本意就是希望模型能够学习到物品的共现模式，我们希望全局物品图中的物品表示能够有助于模型推荐最相关联的物品

作用：为训练数据生成拓扑结构图
输入: 每个物品的序列记录list(原始数据集物品/用户的id从0开始)
输出: 
    train_interaction_list:物品图的邻接表(id从1开始,0用于padding)                 a
    用户物品交互矩阵以及scipy格式的矩阵存储                                         b
    按照L+T切分的列表（优先保证T，不足的L用0填充）
    训练集:
     1)用户的id列表（从0开始）                                                   c
     2)长度为L的物品列表训练用作为输入                                             d
     3)长度为T的物品列表训练用用于预测                                             e
    测试集:
     1) 用户id列表
     2) 长度为L的物品列表训练用作为输入 
     3) 长度为T的物品列表评估模型
                                                   
------------------------------------------------------------------------------------   
创新点1：（全局图的multi-view + contrastive loss)(总体兴趣)  +  局部兴趣  融合预测   (11.7）
    对物品实现全局图的嵌入与用户个性化的局部图的嵌入表示
------------------------------------------------------------------------------------
创新点2： 考虑时间间隔与图模型结合，提出两种时间间隔融合的策略并进行验证分析，发现合理的应用时间间隔在部分场景下有助于提升模型效果
创新点3:  考虑自监督学习应用于提出的模型，通过数据增强+对比学习策略，能够进一步增强稀疏数据中的特征表示，实验表明这种策略是有效的。



HGN的数据集处理方式:
1)按照80%和20%的划分原始数据集为训练集和测试集，对于每个用户，如果有训练集中的物品出现在测试集则将该物品从训练集中去除
2)20%的测试集直接作为训练标签
3)最终模型效果评估时去除让该用户已经交互过的物品评分为0(模型评估去除这个步骤的，效果会差很多)
HGN训练的时候划分的session为L = 5,T = 3

RetaGNN数据集处理方式:
1)按照6:2:2 划分训练/验证/测试集
2) session的划分按照 11 + 3 的格式划分

个人数据处理（遵循HGN）:
1) 20%作为测试集
2) 80%数据作为训练集（训练集中去除测试集中出现的物品）

重复10次实验，取平均结果

SasRec模型的处理方式:
1) 最新的物品用于预测 2）次新的物品用于验证 3）其他都作为训练集

模型评估方式：
1）随机采用100个负样本，然后与1个正样本进行排序，从而计算出NDCG@10以及Hit@10
2）对整个物品集合进行排序（主流方式）


总结：
1）遵循SasRec对于数据集的处理方式（从验证集获取最好的结果）
2）模型验证在整个物品集合上

'''


def preprocess_dataset(dataset_list, userNum, itemNum, input_len=5, target_len=3, item_shift=1, item_g=False,
                       user_g=False):
    print("原始数据集的用户数:%d,物品数目:%d" % (userNum, itemNum))

    # 数据padding,物品id = 0用于padding
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1

    # step1: 根据target_len划分训练集和测试集,使用滑动窗口产生对应的训练集合和测试集合
    train_user_list = []
    train_input_list = []
    train_target_list = []
    split_train_list = []

    test_user_list = []
    test_input_list = []
    test_target_list = []
    train_seq_num = 0
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-target_len])  # 按照target_len划分训练集和测试集
        train_input, train_target = generate_train(item_list[0:-target_len], input_len, target_len)
        for i in range(len(train_input)):
            train_user_list.append(user_id)
            train_input_list.append(train_input[i])
            train_target_list.append(train_target[i])
        train_seq_num += len(train_input)

        test_input, test_target = generate_test(item_list, input_len, target_len)
        test_user_list.append(user_id)
        test_input_list.append(test_input)
        test_target_list.append(test_target)
    assert userNum == len(test_user_list)
    print("训练的序列数目: %d,测试的序列数目: %d" % (train_seq_num, len(test_user_list)))
    train_seq_list = [train_user_list, train_input_list, train_target_list]    # 训练数据[用户id列表，输入序列列表，目标序列列表]
    train_seq_array = [np.array(t) for t in train_seq_list]
    test_seq_list = [test_user_list, test_input_list, test_target_list]        # 测试数据[用户id列表，输入序列列表，目标序列列表]
    test_seq_array = [np.array(t) for t in test_seq_list]

    # step2: 生成用户物品交互矩阵csr
    print("开始生成用户-物品交互矩阵，矩阵大小(%d,%d)" % (userNum, itemNum+item_shift))
    user_item_crs_mat = list_to_matrix(userNum, itemNum+item_shift, split_train_list)
    print("用户物品交互矩阵生成成功")

    # step3: 根据训练集生成物品图的交互列表
    # edge_index: torch.Size([2, edge_num_sum])
    if item_g:
        print("开始生成物品-物品交互矩阵，矩阵大小(%d,%d)" % (itemNum + item_shift, itemNum + item_shift))
        nodeNum = itemNum+item_shift
        global_item_graph_list = generate_adjacency_list_for_item_graph1(split_train_list, 2, nodeNum)
        global_item_graph_csr_mat = list_to_matrix(nodeNum, nodeNum, global_item_graph_list)
        item_edge_index = csr_to_edge_index(global_item_graph_csr_mat)
        global_item_info = [global_item_graph_list, global_item_graph_csr_mat, item_edge_index]
        print("物品-物品交互矩阵生成成功")
    else:
        print("不建立物品-物品图")
        global_item_info = []

    # step4:生成用户-用户全局图------------------------
    if user_g:
        print("开始生成用户-用户交互矩阵，矩阵大小(%d,%d)" % (userNum, userNum))
        nodeNum = userNum
        global_user_graph_list = generate_adjacency_list_for_user_graph1(split_train_list, userNum, itemNum, item_shift)
        global_user_graph_csr_mat = list_to_matrix(nodeNum, nodeNum,  global_user_graph_list)
        user_edge_index = csr_to_edge_index(global_user_graph_csr_mat)
        global_user_info = [global_user_graph_list, global_user_graph_csr_mat, user_edge_index]
    else:
        print("不建立用户-用户图")
        global_user_info = []

    return train_seq_array, test_seq_array, global_item_info, global_user_info, user_item_crs_mat


def preprocess_dataset_new(dataset_list, userNum, itemNum, input_len=5, target_len=3, item_shift=1, item_g=False,
                           user_g=False):
    print("原始数据集的用户数:%d,物品数目:%d" % (userNum, itemNum))

    # 数据padding,物品id = 0用于padding
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1

    # step1: 根据target_len划分训练集和测试集,使用滑动窗口产生对应的训练集合和测试集合
    train_user_list = []
    train_input_list = []
    train_target_list = []
    split_train_list = []

    test_user_list = []
    test_input_list = []
    test_target_list = []
    train_seq_num = 0
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-target_len])  # 按照target_len划分训练集和测试集
        train_input, train_target = generate_train(item_list[0:-target_len], input_len, target_len)
        for i in range(len(train_input)):
            train_user_list.append(user_id)
            train_input_list.append(train_input[i])
            train_target_list.append(train_target[i])
        train_seq_num += len(train_input)

        test_input, test_target = generate_test(item_list, input_len, target_len)
        test_user_list.append(user_id)
        test_input_list.append(test_input)
        test_target_list.append(test_target)
    assert userNum == len(test_user_list)
    print("训练的序列数目: %d,测试的序列数目: %d" % (train_seq_num, len(test_user_list)))
    train_seq_list = [train_user_list, train_input_list, train_target_list]    # 训练数据[用户id列表，输入序列列表，目标序列列表]
    train_seq_array = [np.array(t) for t in train_seq_list]
    test_seq_list = [test_user_list, test_input_list, test_target_list]        # 测试数据[用户id列表，输入序列列表，目标序列列表]
    test_seq_array = [np.array(t) for t in test_seq_list]

    # step2: 生成用户物品交互矩阵csr
    print("开始生成用户-物品交互矩阵，矩阵大小(%d,%d)" % (userNum, itemNum+item_shift))
    user_item_crs_mat = list_to_matrix(userNum, itemNum+item_shift, split_train_list)
    print("用户物品交互矩阵生成成功")

    # step3: 根据训练集生成物品图的交互列表
    # edge_index: torch.Size([2, edge_num_sum])
    if item_g:
        print("开始生成物品-物品交互矩阵，矩阵大小(%d,%d)" % (itemNum + item_shift, itemNum + item_shift))
        nodeNum = itemNum+item_shift
        global_item_graph_list = generate_adjacency_list_for_item_graph1(split_train_list, 2, nodeNum)
        global_item_graph_csr_mat = list_to_matrix(nodeNum, nodeNum, global_item_graph_list)
        item_edge_index = csr_to_edge_index(global_item_graph_csr_mat)
        global_item_info = [global_item_graph_list, global_item_graph_csr_mat, item_edge_index]
        print("物品-物品交互矩阵生成成功")
    else:
        print("不建立物品-物品图")
        global_item_info = []

    # step4:生成用户-用户全局图------------------------
    if user_g:
        print("开始生成用户-用户交互矩阵，矩阵大小(%d,%d)" % (userNum, userNum))
        nodeNum = userNum
        global_user_graph_list = generate_adjacency_list_for_user_graph1(split_train_list, userNum, itemNum, item_shift)
        global_user_graph_csr_mat = list_to_matrix(nodeNum, nodeNum,  global_user_graph_list)
        user_edge_index = csr_to_edge_index(global_user_graph_csr_mat)
        global_user_info = [global_user_graph_list, global_user_graph_csr_mat, user_edge_index]
    else:
        print("不建立用户-用户图")
        global_user_info = []

    return train_seq_array, test_seq_array, global_item_info, global_user_info, user_item_crs_mat


# 将列表按照 input_len+target_len  长度切分
def generate_train(train_list, input_len, target_len):
    size = input_len + target_len
    diff = size - len(train_list)

    # 长度不足进行padding
    if diff > 0:
        tmp_list = [0 for i in range(diff)]
        for v in train_list:
            tmp_list.append(v)
        train_list = tmp_list

    input_list = []
    target_list = []
    for s in range(0, len(train_list) - size + 1):
        input_list.append(train_list[s:s + input_len])
        target_list.append(train_list[s + input_len:s + size])
    return input_list, target_list


def generate_test(item_list, input_len, target_len):
    size = input_len + target_len
    diff = size - len(item_list)

    # 长度不足进行padding
    if diff > 0:
        tmp_list = [0 for i in range(diff)]
        for v in item_list:
            tmp_list.append(v)
        item_list = tmp_list
    input_list = item_list[-size:-target_len]
    target_list = item_list[-target_len:]
    return input_list, target_list


'''
函数说明：根据划分的训练集建立物品-物品图
输入:
    train_list: 划分好的训练集物品序列
    t: 建立边的窗口大小 [pos-t，pos+t+1)
    nodeNum:物品图的节点数量 = 物品数目+paddingNum 
输出:
   无向图的邻接表: 每个节点的邻居关系，在训练集中的邻居关系，即使节点id没有邻居，也需要有一个空列表

物品图的建立策略: 
    对于物品序列中每个物品，将其与前后t个物品建立边
'''


def generate_adjacency_list_for_item_graph1(train_list, t, nodeNum):  # 节点数目 = 物品总数+padding的数目
    # 物品图的邻接表
    adjacency_list = []
    for i in range(nodeNum):
        adjacency_list.append(set())             # 使用集合去除重复的邻居节点
    for tmp_list in train_list:
        for cur in range(len(tmp_list)):
            for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))):
                u = tmp_list[cur]
                v = tmp_list[idx]
                if u == v:                       # 避免self-loop
                    continue
                adjacency_list[u].add(v)
    for i in range(nodeNum):
        adjacency_list[i] = list(adjacency_list[i])
    return adjacency_list


'''
函数说明：根据划分的训练集建立 用户图-用户图
输入: 
    train_list: 划分好的训练集物品序列
    userNum:  用户数目
    itemNUm:  物品数目
    item_shift: 1 ， 0用于padding
输出:
    用户图的邻接表: 
    
基于用户/物品的协同过滤算法:
1）根据已有数据集计算相似度矩阵， wij表示 i和j的相似度
2）利用相似度矩阵预测

用户相似度计算:   公共喜欢的物品的数目/2个用户总的数目
物品相似度的计算: 公共交互过用户的数目/2个物品总的用户数目

这种建图的方式在movieLen数据集上会爆内存
数据集中热们物品会被2万多人给购买，因此通过用户之间存在物品就建立用户的边显然不可靠
如何考虑在热点的物品？ 如何考虑冷们的物品？ 建立一个全局用户图
采用 用户与用户的公共物品个数 >= 阈值 方式建立图，提高阈值能够降低热点物品对用户关系的影响，但一些cold物品之间的u关系将无法在图上体现出来

建立图的方式的探讨：
新的方式： 1)基于用户-用户的公共物品个数建立图，参数不好设置，不同数据集参数未必一致，提高阈值能够降低热点物品对用户关系的影响，但一些cold物品之间的u关系将无法在图上体现出来
         2)基于物品协同过滤的思想，计算用户-用户之间的相似度，然后设置一个阈值，限制每个用户的边的数目
         
'''


def generate_adjacency_list_for_user_graph1(train_list, userNum, itemNum,item_shift):
    # 建立 物品id - 用户列表
    ad_list_for_item = []
    for i in range(itemNum+item_shift):
        ad_list_for_item.append(set())
    for useId, user_item_list in enumerate(train_list):
        for itemId in user_item_list:
            ad_list_for_item[itemId].add(useId)
    print("已经建立好  (key,value)  = (物品id,交互过的用户列表)")
    for i in range(itemNum+item_shift):
        ad_list_for_item[i] = list(ad_list_for_item[i])
    # 用户物品的图的建立方式，购买同一个物品的用户存在边的关系
    user_adj_list = []
    for i in range(userNum):
        user_adj_list.append(set())
    for user_list in ad_list_for_item:
        print(len(user_list))
        # for u in user_list:
        #     for v in user_list:
        #         if u == v:
        #             continue
        #         user_adj_list[u].add(v)
        #         user_adj_list[v].add(u)
    return user_adj_list


# 将邻接表转换为csr格式矩阵
def list_to_matrix(rowNum, colNum, adjacency_list):
    row = []
    col = []
    data = []
    for u, neighbor in enumerate(adjacency_list):
        for v in neighbor:
            row.append(u)
            col.append(v)
            data.append(1)
    # 注意重复的(u,v,data)会造成data的累加
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(rowNum, colNum))
    return adjacency_matrix


# 将csr格式的矩阵转换为(边的数目,2)的numpy数组
def csr_to_edge_index(csr_mat):
    coo_mat = csr_mat.tocoo(copy=True)
    row_index = coo_mat.row
    col_index = coo_mat.col
    edge_index = np.empty(shape=(len(row_index), 2), dtype=int)
    edge_index[:, 0] = row_index
    edge_index[:, 1] = col_index
    return edge_index


class NewDataset(Dataset):
    # 用户id,用户的输入物品列表，用户
    def __init__(self, dataset, cur_index):
        super().__init__()
        self.used_index_list = cur_index
        self.user_ids = dataset[0][cur_index].astype(int)  # 用户id可以用于从负样本数组中获取每个用户的负样本
        self.input_item_array = dataset[1][cur_index].astype(int)
        self.target_item_array = dataset[2][cur_index].astype(int)

    def __getitem__(self, index):  # when dataset[index],this function will be called.
        user_array = self.user_ids[index]
        input_item_array = self.input_item_array[index, :]
        target_item_array = self.target_item_array[index]
        return user_array, input_item_array, target_item_array

    def __len__(self):  # when use len(), this function will be called.
        return len(self.used_index_list)


def get_loader_graph_info_rating_matrix(data_path, batch_size, input_len=5, target_len=3, item_g=False, user_g=False):
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(data_path)
    train, test, item_graph, user_graph, train_user_item_rating = preprocess_dataset(dataset_list=sorted_item_list,
                                                                                     userNum=para_dic['userNum'],
                                                                                     itemNum=para_dic['itemNum'],
                                                                                     input_len=input_len,
                                                                                     target_len=target_len,
                                                                                     item_shift=1,
                                                                                     item_g=False,
                                                                                     user_g=False
                                                                                     )
    train_number = len(train[0])
    test_number = len(test[0])
    print("训练集数目: %d,测试集数目: %d" % (train_number, test_number))
    train_index = [i for i in range(train_number)]
    train_dataset = NewDataset(train, train_index)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_index = [i for i in range(test_number)]
    test_dataset = NewDataset(test, test_index)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True, num_workers=1)

    return train_loader, test_loader, item_graph, user_graph, train_user_item_rating


if __name__ == "__main__":
    data_path_list = ['../DatasetNew/Amazon/preprocess_ratings_Beauty.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Books.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_CDs_and_Vinyl.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Digital_Music.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Movies_and_TV.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Video_Games.pzip',
                      '../DatasetNew/MovieLen/preprocess_ml-10m-ratings.pzip',
                      '../DatasetNew/MovieLen/preprocess_ml-20m-ratings.pzip'
                      ]
    path = data_path_list[6]
    train_l, test_l, _, _, rating_matrix, = get_loader_graph_info_rating_matrix(path, 128, input_len=5, target_len=3)
    for batch_idx, batch_data in enumerate(test_l):
        print(batch_idx)
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print(batch_data[2].shape)
        print(batch_data[0])
        print(type(batch_data[0]))
        print(type(batch_data[1]))
        print(type(batch_data[2]))
        print(batch_data[0].dtype)
        print(batch_data[1].dtype)
        print(batch_data[2].dtype)
        break






