import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
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


def preprocess_dataset(dataset_list, userNum, itemNum, item_shift=1, item_g=False,
                       user_g=False):
    print("原始数据集的用户数:%d,物品数目:%d" % (userNum, itemNum))

    # 数据padding,物品id = 0用于padding(原始数据的物品id是从0开始的)
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1
            if len(item_list) < 5:
                print(len(item_list))
                print("yes")

    # step1: 用户的购买序列list
    user_seq = dataset_list
    split_train_list = []
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-2])  # 每个用户序列分别保留两个物品分别作为 验证集 和 测试集
    assert userNum == len(user_seq)

    # step2: 生成用户-物品评分矩阵(验证集/测试集)
    print("开始生成用户-物品交互矩阵，矩阵大小(%d,%d)" % (userNum, itemNum + item_shift))
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, userNum, itemNum + item_shift)
    test_rating_matrix = generate_rating_matrix_test(user_seq, userNum, itemNum + item_shift)
    print("用户物品交互矩阵生成成功")
    valid_test_rating_matrix = [valid_rating_matrix, test_rating_matrix]

    # step3: 根据训练集生成物品图的交互列表
    # edge_index: torch.Size([2, edge_num_sum])
    if item_g:
        print("开始生成物品-物品交互矩阵，矩阵大小(%d,%d)" % (itemNum + item_shift, itemNum + item_shift))
        nodeNum = itemNum + item_shift
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
        global_user_graph_csr_mat = list_to_matrix(nodeNum, nodeNum, global_user_graph_list)
        user_edge_index = csr_to_edge_index(global_user_graph_csr_mat)
        global_user_info = [global_user_graph_list, global_user_graph_csr_mat, user_edge_index]
    else:
        print("不建立用户-用户图")
        global_user_info = []

    return user_seq, valid_test_rating_matrix, global_item_info, global_user_info


# 验证集中剔除最后2个物品，让其他所有物品来构建用户-物品评分矩阵
def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


# 测试集中剔除最后一个物品，让其余所有物品构建用户物品的评分矩阵
def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


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
        adjacency_list.append(set())  # 使用集合去除重复的邻居节点
    for tmp_list in train_list:
        for cur in range(len(tmp_list)):
            for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))):
                u = tmp_list[cur]
                v = tmp_list[idx]
                if u == v:  # 避免self-loop
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
         2)基于物品协同过滤的思想，计算用户-用户之间的相似度，提高
         3)参考TiSasRec，对时间间隔进行处理，然后在建立物品之间的边的时候，对于比较长的时间间隔不考虑
用户图不是太靠谱啊
'''


def generate_adjacency_list_for_user_graph1(train_list, userNum, itemNum, item_shift):
    # 建立 物品id - 用户列表
    ad_list_for_item = []
    for i in range(itemNum + item_shift):
        ad_list_for_item.append(set())
    for useId, user_item_list in enumerate(train_list):
        for itemId in user_item_list:
            ad_list_for_item[itemId].add(useId)
    print("已经建立好  (key,value)  = (物品id,交互过的用户列表)")
    for i in range(itemNum + item_shift):
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


class MySASRecDataset(Dataset):
    def __init__(self, user_seq, item_num, max_seq_length, data_type='train'):
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = max_seq_length
        self.item_num = item_num

    '''
        [0, 1, 2, 3, 4, 5, 6]
        
        train [0, 1, 2, 3]
        target [1, 2, 3, 4],       模型的正样本
            0      正样本是 1
            0,1    正样本是 2
            0,1,2  正样本是 3
            0,1,2,4 正样本是 4
            
        valid [0, 1, 2, 3, 4]
        answer [5]

        test [0, 1, 2, 3, 4, 5]
        answer [6]
    '''

    def __getitem__(self, index):
        user_id = index
        # 注意这里index与user_id一致，需要设置dataloader的RandomSampler为随机不重复采样，而不是shuffle = true + 顺序采样
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # s1:划分数据集
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        # s2:采集训练时需要的负样本
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_num))

        # s3:对数据使用0进行padding,确保数据统一长度
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),  # 输入的物品的id
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def __len__(self):
        return len(self.user_seq)


# [1,item_size] and 不包含在item_set中
def neg_sample(item_set, item_size):
    item = random.randint(1, item_size)
    while item in item_set:
        item = random.randint(1, item_size)
    return item


def get_loader_graph_info_rating_matrix(data_path, batch_size, max_seq_len, item_g=False, user_g=False):
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(data_path)
    user_seq, valid_test_rating_matrix, global_item_info, global_user_info = preprocess_dataset(
        dataset_list=sorted_item_list,
        userNum=para_dic['userNum'],
        itemNum=para_dic['itemNum'],
        item_shift=1,
        item_g=False,
        user_g=False
    )
    dataset_list = [MySASRecDataset(user_seq, item_num=para_dic['itemNum'], max_seq_length=max_seq_len, data_type=t)
                    for t in ["train", "valid", "test"]]
    # 验证集和测试集不需要打乱
    sampler_list = [RandomSampler(dataset_list[0]), SequentialSampler(dataset_list[1]), SequentialSampler(dataset_list[2])]
    loader_list = []
    for i in range(3):
        loader_list.append(DataLoader(dataset=dataset_list[i], batch_size=batch_size, sampler=sampler_list[i], num_workers=6,
                                      pin_memory=True))
    return loader_list, global_item_info, global_user_info, valid_test_rating_matrix


if __name__ == "__main__":
    data_path_list = ['../DatasetNew/Amazon/preprocess_ratings_Beauty.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Books.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_CDs_and_Vinyl.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Digital_Music.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Movies_and_TV.pzip',
                      '../DatasetNew/Amazon/preprocess_ratings_Video_Games.pzip',
                      '../DatasetNew/MovieLen/preprocess_ml-10m-ratings.pzip',
                      '../DatasetNew/MovieLen/preprocess_ml-20m-ratings.pzip',
                      '../DatasetNew/HIN_Dataset/preprocess_my_yelp.pzip'
                      ]
    path = data_path_list[8]
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(path)
    print(para_dic)
    v = 100000000
    for item_list in sorted_item_list:
        v = min(v, len(item_list))
    print(v)

    data_loader_list, _, _, rating_matrix, = get_loader_graph_info_rating_matrix(path, 128, max_seq_len=50)
    train_loader = data_loader_list[2]
    for batch_idx, batch_data in enumerate(train_loader):
        print(batch_idx)
        # print(batch_data[0].shape)
        # print(batch_data[1].shape)
        # print(batch_data[2].shape)
        # print(batch_data[3].shape)
        # print(batch_data[4].shape)
        # print(batch_data[0])
        # print(batch_data[0].dtype)
        # print(batch_data[1].dtype)
        # print(batch_data[2].dtype)


    # target_neg = [0] * -1 + [1, 2, 3]
    # print(target_neg)
    # target_neg = [0] * 1 + [1, 2, 3]
    # print(target_neg)
    # target_neg = [0] * 5 + [1, 2, 3]
    # print(target_neg)
