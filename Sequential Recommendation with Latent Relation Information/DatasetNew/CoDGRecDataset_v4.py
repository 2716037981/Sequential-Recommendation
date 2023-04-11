import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from util.myutil import *
from scipy.sparse import csr_matrix
from DatasetNew.GraphBuilder import *
from DatasetNew.CoDGRec_data_augmentation import *
from models.MyModel1 import OnlineItemSimilarity


import numpy as np

# parser = argparse.ArgumentParser()
# # data arguments
# parser.add_argument('--dir', type=int, default=3)
#
# # train arguments
# parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
# parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
# parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
# parser.add_argument("--seed", default=999, type=int)
#
# # learning related
# parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
# parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
# parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
#
# # model dependent arguments
# parser.add_argument("--model_name", default='CoDGRec', type=str)
# parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
# parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
# parser.add_argument('--num_attention_heads', default=2, type=int)
# parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
# parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
# parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
# parser.add_argument("--initializer_range", type=float, default=0.02)
# parser.add_argument('--max_seq_length', default=50, type=int)
# # - global item graph parameters
# parser.add_argument('--item_GNN_layer_num', default=2, type=int)
# parser.add_argument('--item_GNN_dropout', default=0.5, type=float)
# # - global user graph parameters
# parser.add_argument('--user_GNN_layer_num', default=2, type=int)
# parser.add_argument('--user_GNN_dropout', default=0.5, type=float)
# # - item/user graph build hyperparameters
# parser.add_argument('--t', default=2, type=int)  # 窗口的大小
# parser.add_argument('--item_graph_span', default=128, type=int)  # 时间戳嵌入的范围
# parser.add_argument("--item_neighbor_num", default=100, type=int)  # 限制每个物品的邻居节点数量
# parser.add_argument("--user_neighbor_num", default=5, type=int)  # 限制每个用户的邻居节点数量
# parser.add_argument("--hot_item_limit", default=100, type=int)  # 用户建图协同过滤算法的共现矩阵不考虑超热点物品（提高效率）
#
# # 探究权重对于模型的影响
# parser.add_argument('--info_use', default=2, type=int, help="0:use item frequency,1:use time interval,2:use both ")
# # 探究关联信息对于模型的影响
# parser.add_argument('--user_rel', default=True, type=bool, help="whether use information from user relationship")
# parser.add_argument('--item_rel', default=False, type=bool, help="whether use information from item relationship")
#
# # ==========================================数据增强相关参数（insert和substitute非常耗费时间==================================
# # 长短序列划分阈值
# parser.add_argument('--augment_threshold', default=4, type=int)
# # 相似度计算算法：Random, ItemCF，ItemCF_IUF(Inverse user frequency), Item2Vec
# parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str)
# # 相似度融合策略
# parser.add_argument('--similarity_model_type', default='offline', type=str, help="offline,online,hybrid")
# # 相似度切换的epoch数目
# parser.add_argument("--augmentation_warm_up_epoches", type=float, default=40)
# # 序列数据增强的方法： mask, crop, reorder, substitute, insert, random，combinatorial_enumerate
# parser.add_argument('--base_augment_type', default="crop", type=str)
# # 采用Random的方式进行数据增强，短序列所使用的数据增强方法：SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC
# parser.add_argument('--augment_type_for_short', default=7, type=int)
#
# # =====================================5种序列数据增强方式的超参数设置=====================================
# # 随机选择连续的子序列，控制裁剪的子序列的长度: tao * n
# parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
# # 随机mask子序列中固定数量的物品: 控制mask物品的数量: gamma * n
# parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
# # 随机选择连续的子序列然后随机打乱顺序: 控制子序列的长度: beta * n
# parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")
# # 随机选择序列中k个物品进行替换: 替换的物品数量 = n * rate，论文中的数值设置[0.1,0.9]
# parser.add_argument("--substitute_rate", type=float, default=0.1)
# # 随机选择序列中k个位置的物品，在其后面插入最相似的物品: 插入物品的数量 = rate * n，论文中的数值设置[0.1,0.9]
# parser.add_argument("--insert_rate", type=float, default=0.4, help="insert ratio for insert operator")
# # 每个物品最多插入多少个关联物品
# parser.add_argument("--max_insert_num_per_pos", type=int, default=1)
#
# # =====================================训练loss相关参数================================
# # contrastive learning task args
# parser.add_argument('--temperature', default=1.0, type=float, help='softmax temperature (default:  1.0) - not studied.')
# # 对比学习loss的超参数，论文中设置为{0.1,0.2,0.3,0.4,0.5}
# parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
# parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")


def preprocess_dataset_v1(dataset_list, timestamp_list, para_dic, args, item_shift=1):
    print("user number:%d,item number:%d" % (para_dic['userNum'], para_dic['itemNum']))

    # 数据padding,物品id = 0用于padding(原始数据的物品id是从0开始的)
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1

    # step1: 用户的购买序列list
    user_seq = dataset_list
    split_train_list = []
    split_time_list = []
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-2])  # 每个用户序列分别保留两个物品分别作为 验证集 和 测试集
        split_time_list.append(timestamp_list[user_id][0:-2])
    assert para_dic['userNum'] == len(user_seq)

    # step2: 生成用户-物品评分矩阵(验证集/测试集)
    userNum = para_dic['userNum']
    itemNum = para_dic['itemNum']
    print("Generate user-item rating matrix:(%d,%d)" % (userNum, itemNum + item_shift))
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, userNum, itemNum + item_shift)
    test_rating_matrix = generate_rating_matrix_test(user_seq, userNum, itemNum + item_shift)
    print("Finish generating!")
    valid_test_rating_matrix = [valid_rating_matrix, test_rating_matrix]

    print("Item graph builder:strategy 4!")
    item_node_num = para_dic['itemNum'] + item_shift
    # [边，频率权重，时间权重]
    item_info_list = MyGraph(ItemGraphBuilder4()).make_graph(item_node_num, split_train_list, split_time_list, args)

    print("User graph builder:strategy 1!")
    user_node_num = para_dic['userNum']
    user_info_list = MyGraph(UserGraphBuilder1()).make_graph(user_node_num, split_train_list, split_time_list, args)

    return user_seq, valid_test_rating_matrix, item_info_list, user_info_list


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


class CoDGRecDataset(Dataset):
    def __init__(self, user_seq, item_num, max_seq_length,args,data_type='train', similarity_model_type='offline',
                 base_augment_type="random"):
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = max_seq_length
        self.item_num = item_num
        self.n_views = 2  # 默认是2就足够
        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        if similarity_model_type == 'offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type == 'online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type == 'hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(tao=args.tao),
                              'mask': Mask(gamma=args.gamma),
                              'reorder': Reorder(beta=args.beta),
                              'substitute': Substitute(self.similarity_model,
                                                       substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model,
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': Random(tao=args.tao, gamma=args.gamma,
                                               beta=args.beta, item_similarity_model=self.similarity_model,
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos,
                                               substitute_rate=args.substitute_rate,
                                               augment_threshold=args.augment_threshold,
                                               augment_type_for_short=args.augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(tao=args.tao, gamma=args.gamma,
                                                                                beta=args.beta,
                                                                                item_similarity_model=self.similarity_model,
                                                                                insert_rate=args.insert_rate,
                                                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                                                substitute_rate=args.substitute_rate,
                                                                                n_views=2)
                              }
        if base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[base_augment_type]

    '''
       输入:训练集的物品序列
       输出：每个序列增强的物品的物品
    '''

    def _one_pair_data_augmentation(self, input_ids):
        """
            provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            # 核心函数base_transform:对物品序列进行增强
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = (
                torch.tensor(augmented_input_ids, dtype=torch.long)
            )
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

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
            # 对原始用户序列进行数据增强
            cf_tensors_list = []
            total_augmentaion_pairs = nCr(self.n_views, 2)  # 计算C_{n}^{2} = 1
            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))
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

        # 对数据使用0进行padding,确保数据统一长度
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
        if self.data_type == "train":
            return cur_rec_tensors, cf_tensors_list
        else:
            return cur_rec_tensors

    def __len__(self):
        return len(self.user_seq)


# [1,item_size] and 不包含在item_set中
def neg_sample(item_set, item_size):
    item = random.randint(1, item_size)
    while item in item_set:
        item = random.randint(1, item_size)
    return item


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)  # n! // r! // (n-r)!,计算c_{n}^{r}


def get_loader_graph_info_rating_matrix_v1(data_path, batch_size, max_seq_len, args):
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(data_path)
    user_seq, valid_test_rating_matrix, global_item_info, global_user_info = preprocess_dataset_v1(
        dataset_list=sorted_item_list,
        timestamp_list=sorted_time_list,
        para_dic=para_dic,
        args=args,
        item_shift=1
    )
    '''
    # 定义相似度计算方法，这里离线相似度的计算比较快
    # 在线相似度需要每个epoch开始时存储每个物品的嵌入向量最相似的向量，物品数目多时比较耗费时间（待实现）
    '''
    data_name = data_path.split("/")[-1].split(".")[0]
    offline_similarity_model = OfflineItemSimilarity(seq_list=sorted_item_list,
                                                     similarity_path="nothing",
                                                     model_name=args.similarity_model_name,
                                                     dataset_name=data_name)
    args.offline_similarity_model = offline_similarity_model

    # 定义数据集，需要注意的是这里只有训练数据集需要进行数据增强
    train_set = CoDGRecDataset(user_seq, item_num=para_dic['itemNum'], args=args, max_seq_length=max_seq_len, data_type="train",
                               similarity_model_type=args.similarity_model_type,
                               base_augment_type=args.base_augment_type)
    valid_set = CoDGRecDataset(user_seq, item_num=para_dic['itemNum'], max_seq_length=max_seq_len, args=args, data_type="valid",)
    test_set = CoDGRecDataset(user_seq, item_num=para_dic['itemNum'], max_seq_length=max_seq_len, args=args, data_type="test")
    dataset_list = [train_set, valid_set, test_set]

    # 验证集和测试集不需要打乱
    sampler_list = [RandomSampler(dataset_list[0]), SequentialSampler(dataset_list[1]),
                    SequentialSampler(dataset_list[2])]
    loader_list = []
    for i in range(3):
        loader_list.append(
            DataLoader(dataset=dataset_list[i], batch_size=batch_size, sampler=sampler_list[i], num_workers=6,
                       pin_memory=False))
    return loader_list, global_item_info, global_user_info, valid_test_rating_matrix


# =====================================Z
class OfflineItemSimilarity:
    def __init__(self, seq_list=None, similarity_path=None, model_name='ItemCF',
                 dataset_name='Sports_and_Outdoors'):
        self.model_name = model_name

        # 存储每个物品最相似的物品以及得分
        path = "../DatasetNew/similarity_item_path/" + dataset_name + "_item_similarity.pzip"
        self.most_similar_item_list = {}
        self.most_similar_item_list_with_score = {}
        try:
            sim_list = load_decompressed_pzip(path)
        except Exception:
            print("build item sim information Now!")
            _, _, train_data_dict = self._load_my_train_data(seq_list)
            # 获取相似度矩阵
            item_sim_dict = self._generate_item_similarity(train_data_dict)
            # 计算相似度矩阵中相似度的最大/最小值
            self.max_score, self.min_score = -1, 100
            for item in item_sim_dict.keys():
                for neig in item_sim_dict[item]:
                    sim_score = item_sim_dict[item][neig]
                    self.max_score = max(self.max_score, sim_score)
                    self.min_score = min(self.min_score, sim_score)
            # 计算每个物品最相似的物品
            for item in item_sim_dict.keys():
                # 只考虑最相似的物品
                items_with_score = sorted(item_sim_dict[item].items(), key=lambda x: x[1], reverse=True)[0:1]
                self.most_similar_item_list[item] = list(map(lambda x: int(x[0]), items_with_score))
                self.most_similar_item_list_with_score[item] = list(
                    map(lambda x: (int(x[0]), (self.max_score - float(x[1]))
                                   / (self.max_score - self.min_score)), items_with_score))
            # 存储物品
            store_compressed_pzip([self.most_similar_item_list, self.most_similar_item_list_with_score], path)
        else:
            self.most_similar_item_list = sim_list[0]
            self.most_similar_item_list_with_score = sim_list[1]

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    # 自己数据集处理
    def _load_my_train_data(self, seq_list):
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for idx, item_list in enumerate(seq_list):
            train_data_list.append(item_list[:-3])
            train_data_set_list += item_list
            for item_id in item_list:
                train_data.append((idx + 1, item_id, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            return self.itemSimBest

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec']:
            """TODO: handle case that item not in keys(相似度矩阵中没有该物品则返回原物品)"""
            if str(item) in self.most_similar_item_list:
                return self.most_similar_item_list[item]
            elif int(item) in self.most_similar_item_list:
                if with_score:
                    return self.most_similar_item_list_with_score[item]
                return self.most_similar_item_list[item]
            else:
                return [item for t in range(top_k)]


# if __name__ == "__main__":
#     args = parser.parse_args()
#     data_path_list = [
#         '../DatasetNew/Amazon/preprocess_ratings_Digital_Music.pzip',
#         '../DatasetNew/Amazon/preprocess_ratings_Video_Games.pzip',
#         '../DatasetNew/MovieLen/preprocess_ml-10m-ratings.pzip',
#         '../DatasetNew/Amazon/preprocess_ratings_Beauty.pzip',
#         '../DatasetNew/MovieLen/preprocess_ml-20m-ratings.pzip',
#         '../DatasetNew/Amazon/preprocess_ratings_Movies_and_TV.pzip',
#         '../DatasetNew/Amazon/preprocess_ratings_CDs_and_Vinyl.pzip',
#         '../DatasetNew/HIN_Dataset/preprocess_my_yelp.pzip',
#         '../DatasetNew/HIN_Dataset/preprocess_my_LastFM.pzip',
#     ]
#     path = data_path_list[8]
#     para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(path)
#     data_loader_list, item_g_info, user_g_info, rating_matrix = get_loader_graph_info_rating_matrix_v1(path,
#                                                                                                        batch_size=args.batch_size,
#                                                                                                        max_seq_len=50,
#                                                                                                       args=args)
#     train_loader = data_loader_list[0]
#     print(para_dic)
#     for epoch in range(10):
#         for a, b in train_loader:
#             print(len(a))
#             print(len(b))
#             print(b[0][0].shape)
#             print(b[0][1].shape)

