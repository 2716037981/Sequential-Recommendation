import torch
from torch.utils.data import Dataset, DataLoader
from util.myutil import *
from scipy.sparse import csr_matrix
from DatasetNew.GraphBuilder import *
import numpy as np

'''
   1) 输入序列 T = 5,L = 1
   2) 划分训练集，验证集，测试集
   3) 

'''

parser = argparse.ArgumentParser()

# data arguments
parser.add_argument('--L', type=int, default=5)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--P', type=int, default=1, help="the number of items in low-order associations")
parser.add_argument('--dir', type=int, default=3)
parser.add_argument('--model_name', type=str, default="ham_new")

# train arguments
parser.add_argument('--n_iter', type=int, default=300)
parser.add_argument('--setting', type=str, default='CUT')
parser.add_argument('--isTrain', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--order', type=int, default=2,
                    help="the order of the item synergies")
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--sets_of_neg_samples', type=int, default=50)

# model dependent arguments
parser.add_argument('--d', type=int, default=50)

# - global item graph parameters
parser.add_argument('--item_GNN_layer_num', default=2, type=int)
parser.add_argument('--item_GNN_dropout', default=0.5, type=float)
# - global user graph parameters
parser.add_argument('--user_GNN_layer_num', default=2, type=int)
parser.add_argument('--user_GNN_dropout', default=0.5, type=float)
# - item/user graph build hyperparameters
parser.add_argument('--t', default=2, type=int)  # 窗口的大小
parser.add_argument('--item_graph_span', default=128, type=int)  # 时间戳嵌入的范围
parser.add_argument('--use_stamp', default=False, type=bool)  # 是否使用时间戳
parser.add_argument('--use_item_freq_use', default=False, type=bool)  # 是否统计物品对出现的频率
parser.add_argument("--item_neighbor_num", default=10, type=int)  # 限制每个物品的邻居节点数量
parser.add_argument("--user_neighbor_num", default=10, type=int)  # 限制每个用户的邻居节点数量
parser.add_argument("--hot_item_limit", default=1000, type=int)  # 用户建图协同过滤算法的共现矩阵不考虑超热点物品（提高效率）
config = parser.parse_args()


def preprocess_dataset_new(dataset_list, timestamp_list, para_dic, args, item_shift=1):
    input_len = args.L
    target_len = args.T
    userNum = para_dic['userNum']
    itemNum = para_dic['itemNum']
    print("原始数据集的用户数:%d,物品数目:%d" % (userNum, itemNum))

    # 数据padding,物品id = 0用于padding
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1

    # 根据input_len/target_len划分训练集和验证集和测试集
    train_user_list = []
    train_input_list = []
    train_target_list = []
    split_train_list = []
    split_train_time_list = []

    valid_user_list = []
    valid_input_list = []
    valid_target_list = []
    split_valid_list = []

    test_user_list = []
    test_input_list = []
    test_target_list = []
    train_seq_num = 0
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-(2 * target_len)])
        split_train_time_list.append(timestamp_list[user_id][0:-(2 * target_len)])
        split_valid_list.append(item_list[0:-target_len])

        train_input, train_target = generate_train(item_list[0:-(2 * target_len)], input_len, target_len)
        for i in range(len(train_input)):
            train_user_list.append(user_id)
            train_input_list.append(train_input[i])
            train_target_list.append(train_target[i])
        train_seq_num += len(train_input)

        valid_input, valid_target = generate_valid_and_test(item_list[0:-target_len], input_len, target_len)
        valid_user_list.append(user_id)
        valid_input_list.append(valid_input)
        valid_target_list.append(valid_target)

        test_input, test_target = generate_valid_and_test(item_list, input_len, target_len)
        test_user_list.append(user_id)
        test_input_list.append(test_input)
        test_target_list.append(test_target)

    assert userNum == len(test_user_list)
    print("训练的序列数目: %d,测试/验证的序列数目: %d" % (train_seq_num, len(test_user_list)))
    train_seq_list = [train_user_list, train_input_list, train_target_list]  # 训练数据[用户id列表，输入序列列表，目标序列列表]
    train_seq_array = [np.array(t) for t in train_seq_list]
    valid_seq_list = [valid_user_list, valid_input_list, valid_target_list]  # 测试数据[用户id列表，输入序列列表，目标序列列表]
    valid_seq_array = [np.array(t) for t in valid_seq_list]
    test_seq_list = [test_user_list, test_input_list, test_target_list]  # 测试数据[用户id列表，输入序列列表，目标序列列表]
    test_seq_array = [np.array(t) for t in test_seq_list]
    seq_info = [train_seq_array, valid_seq_array, test_seq_array]

    # 生成用户-物品评分矩阵(验证集/测试集)
    print("Generate user-item rating matrix:(%d,%d)" % (userNum, itemNum + item_shift))
    valid_rating_matrix = generate_rating_matrix_valid(dataset_list, userNum, itemNum + item_shift)
    test_rating_matrix = generate_rating_matrix_test(dataset_list, userNum, itemNum + item_shift)
    print("Finish generating!")
    valid_test_rating_matrix = [valid_rating_matrix, test_rating_matrix]

    print("Item graph builder:strategy 4!")
    item_node_num = para_dic['itemNum'] + item_shift
    item_info_list = MyGraph(ItemGraphBuilder4()).make_graph(item_node_num, split_train_list, split_train_time_list,
                                                             args)

    print("User graph builder:strategy 1!")
    user_node_num = para_dic['userNum']
    user_info_list = MyGraph(UserGraphBuilder1()).make_graph(user_node_num, split_train_list, split_train_time_list,
                                                             args)

    return seq_info, valid_test_rating_matrix, item_info_list, user_info_list


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


def generate_valid_and_test(item_list, input_len, target_len):
    size = input_len + target_len
    diff = size - len(item_list)

    # 长度不足进行padding
    if diff > 0:
        tmp_list = [0 for _ in range(diff)]
        for v in item_list:
            tmp_list.append(v)
        item_list = tmp_list
    input_list = item_list[-size:-target_len]
    target_list = item_list[-target_len:]
    return input_list, target_list


# 验证集中剔除最后T个物品，让其他所有物品来构建用户-物品评分矩阵
def generate_rating_matrix_valid(user_seq, num_users, num_items, t=1):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2 * t]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


# 测试集中剔除最后一个物品，让其余所有物品构建用户物品的评分矩阵
def generate_rating_matrix_test(user_seq, num_users, num_items, t=1):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-t]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


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


def get_loader_graph_info_rating_matrix(data_path, args):
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(data_path)
    data_info, rating_matrix_info, item_g_info, user_g_info = preprocess_dataset_new(
        dataset_list=sorted_item_list,
        timestamp_list=sorted_time_list,
        para_dic=para_dic,
        args=args,
        item_shift=1
        )
    train_number = len(data_info[0][0])
    test_number = len(data_info[2][0])
    print("训练集数目: %d,测试集数目: %d" % (train_number, test_number))

    train_index = [i for i in range(train_number)]
    train_dataset = NewDataset(data_info[0], train_index)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    valid_index = [i for i in range(test_number)]
    valid_dataset = NewDataset(data_info[1], valid_index)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=256, shuffle=False, num_workers=1)

    test_index = [i for i in range(test_number)]
    test_dataset = NewDataset(data_info[2], test_index)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=1)
    load_list = [train_loader, valid_loader, test_loader]
    return load_list, rating_matrix_info, item_g_info, user_g_info


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
    path = data_path_list[0]
    load_info, rating_info, item_graph_info, user_graph_info = get_loader_graph_info_rating_matrix(path, config)
    for batch_idx, batch_data in enumerate(load_info[0]):
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
    print(str(config))
