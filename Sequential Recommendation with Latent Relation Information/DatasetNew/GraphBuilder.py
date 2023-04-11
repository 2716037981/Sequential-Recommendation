import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from util.myutil import *
from scipy.sparse import csr_matrix
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='SasRec', type=str)
parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument('--num_attention_heads', default=2, type=int)
parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument('--max_seq_length', default=50, type=int)
# - global item graph parameters
parser.add_argument('--item_GNN_layer_num', default=1, type=int)
parser.add_argument('--item_GNN_dropout', default=0.5, type=float)
# - item graph build hyperparameters
parser.add_argument('--t', default=1, type=int)  # 窗口的大小
parser.add_argument('--item_graph_span', default=128, type=int)  # 时间戳嵌入的范围
parser.add_argument('--use_stamp', default=False, type=bool)  # 是否使用时间戳
parser.add_argument('--use_item_freq_use', default=False, type=bool)  # 是否统计物品对出现的频率
parser.add_argument("--item_neighbor_num", default=10, type=int)  # 限制每个物品的邻居节点数量
parser.add_argument("--user_neighbor_num", default=10, type=int)  # 限制每个用户的邻居节点数量
parser.add_argument("--hot_item_limit", default=100, type=int)  # 用户建图协同过滤算法的共现矩阵不考虑超热点物品（提高效率）

'''
物品图建立的4种策略：
1）根据窗口t建立边
2）根据窗口t建立边，然后统计物品对出现次数，对于每个物品选取出现频率高的物品的添加边，选取的物品数目不超过item_neighbor_num
3) 根据窗口t建立边, 并对边嵌入时间，重复出现的边取均值，超过阈值边则舍弃(需要看一下有那些可以嵌入考)
4) 根据窗口t建立边，对于每个物品统计邻居节点的出现频率以及时间差的均值，首先去除那些超过时间阈值的节点，然后再选出不超过item_neighbor_num的节点

用户图的建立策略:
1) 根据用户协同过滤算法，为每个用户选出不超过10个的相似读最高的10个用户

  动机：相似的用户中具有相似的行为模式，
  通过用户图的建立，加强序列的特征表示
  训练方式上:采用多任务学习
  局部的用户嵌入表示+序列嵌入表示 × 目标物品  （评分预测任务）
  用户嵌入表示 + 序列表示 => 记忆网络  => 兴趣表示
  聚合的用户嵌入 + 序列表示 => 记忆网络(参考MAGNN定义m个记忆单元去获取其他相似用户的潜在序列模型) => 兴趣表示
  --------------------------------------------
  用户图建立动机：从用户角度让单个用户能够学习到相似用户的行为依赖
  物品图建立动机: 学习物品的之间的关联信息，用于物品的预测
  
  融合机制:通道注意力机制，对无用信息进行过滤
'''


# 采用策略模型抽象出图的构建算法
class GraphRecBuild(object):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        pass


class ItemGraphBuilder1(GraphRecBuild):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        print("Item graph strategy 1:")
        t = args.t  # 物品图的范围
        # 物品图的邻接表
        adjacency_list = []
        for i in range(node_num):
            adjacency_list.append(set())  # 使用集合去除重复的邻居节点
        for tmp_list in item_list:
            for cur in range(len(tmp_list)):
                for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))):
                    u = tmp_list[cur]
                    v = tmp_list[idx]
                    if u == v:  # 避免self-loop
                        continue
                    adjacency_list[u].add(v)
        edge_sum = 0
        for i in range(node_num):
            adjacency_list[i] = list(adjacency_list[i])
            edge_sum += len(adjacency_list[i])
        print("Edge number of item-item graph: %d" % edge_sum)
        edge_index, edge_weight = list_to_geometric_format(adjacency_list)
        return [edge_index, edge_weight]


class ItemGraphBuilder2(GraphRecBuild):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        t = args.t  # 物品图的范围
        limit = args.item_neighbor_num  # 限制物品的数量

        adjacency_list = []
        for i in range(node_num):
            adjacency_list.append(dict())  # 使用集合去除重复的邻居节点
        for tmp_list in item_list:
            for cur in range(len(tmp_list)):
                u = tmp_list[cur]
                for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))):
                    v = tmp_list[idx]
                    if u == v:  # 避免self-loop
                        continue
                    if v in adjacency_list[u]:
                        adjacency_list[u][v] = adjacency_list[u][v] + 1
                    else:
                        adjacency_list[u][v] = 1

        filter_adjacency_list = []
        filter_adjacency_weight_list = []
        for i in range(node_num):
            filter_adjacency_list.append([])
            filter_adjacency_weight_list.append([])

        edge_num_before = 0
        edge_num_after = 0
        for i in range(node_num):
            adjacency_list[i] = adjacency_list[i].items()
            edge_num_before += len(adjacency_list[i])
            adjacency_list[i] = sorted(adjacency_list[i], reverse=True, key=lambda item: item[1])  # 按照频率排序
            for j in range(min(limit, len(adjacency_list[i]))):  # 限制邻居节点数量
                filter_adjacency_list[i].append(adjacency_list[i][j][0])  # 邻居节点id
                filter_adjacency_weight_list[i].append(adjacency_list[i][j][1])  # 边的权重(出现频率)
            edge_num_after += len(filter_adjacency_list[i])

        print("Total edge number %d,edge number after filtering %d with neighbor number limit %d" %
              (edge_num_before, edge_num_after, limit))
        edge_index, edge_weight = list_to_geometric_format(filter_adjacency_list, filter_adjacency_weight_list)
        return [edge_index, edge_weight]


class ItemGraphBuilder3(GraphRecBuild):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        t = args.t  # 物品图的范围
        limit = args.item_neighbor_num  # 限制物品的数量
        time_span = args.item_graph_span  # 有效时间戳范围

        # 对时间戳列表进行个性化的normalized处理
        scaled_time_list, max_time, min_time = stamp_to_int(sorted_time_list)

        # 获取边，不同用户中重复出现的边则span相加
        adjacency_list = []
        interval_weight_list = []
        for i in range(node_num):
            adjacency_list.append(dict())
            interval_weight_list.append(dict())
        for i in range(len(item_list)):
            tmp_list = item_list[i]
            tmp_time_list = scaled_time_list[i]
            for cur in range(len(tmp_list)):
                u = tmp_list[cur]
                u_time = tmp_time_list[cur]
                for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))):
                    v = tmp_list[idx]
                    v_time = tmp_time_list[idx]
                    if u == v:  # 避免self-loop
                        continue
                    if v in adjacency_list[u]:
                        adjacency_list[u][v] = adjacency_list[u][v] + 1
                        interval_weight_list[u][v] = adjacency_list[u][v] + abs(u_time - v_time)
                    else:
                        adjacency_list[u][v] = 1
                        interval_weight_list[u][v] = abs(u_time - v_time)

        filter_adjacency_list = []
        filter_adjacency_freq_list = []
        filter_adjacency_time_list = []
        for i in range(node_num):
            filter_adjacency_list.append([])
            filter_adjacency_freq_list.append([])
            filter_adjacency_time_list.append([])

        # 通过时间间隔进行过滤
        edge_num_before = 0
        edge_num_after = 0
        two_freq_edge_num = 0
        origin_two_freq_edge_num = 0

        for i in range(node_num):
            cur_item_list = list(adjacency_list[i].items())
            edge_num_before += len(cur_item_list)
            # adjacency_list[i] = sorted(adjacency_list[i], reverse=True, key=lambda item: item[1])  # 按照频率排序
            for j in range(len(cur_item_list)):  # 限制邻居节点数量
                v = cur_item_list[j][0]
                freq = cur_item_list[j][1]
                mean_time_span = interval_weight_list[i][v] / freq
                if freq >= 2:
                    origin_two_freq_edge_num = origin_two_freq_edge_num + 1
                if mean_time_span == 0:
                    mean_time_span = 1
                elif mean_time_span > time_span:
                    mean_time_span = 0
                if mean_time_span != 0:  # 通过平均时间间隔进行过滤
                    filter_adjacency_list[i].append(v)  # 邻居节点id
                    filter_adjacency_freq_list[i].append(freq)  # 边的权重(出现频率)
                    if freq >= 2:
                        two_freq_edge_num = two_freq_edge_num + 1
                    filter_adjacency_time_list[i].append(mean_time_span)
            edge_num_after += len(filter_adjacency_list[i])

        print("Total edge number %d,edge number after filtering %d with time span %d"
              % (edge_num_before, edge_num_after, time_span))
        print("Directed Edge number >= 2 without time span % d,with time span: %d"
              % (origin_two_freq_edge_num, two_freq_edge_num))
        edge_index, freq_edge_weight = list_to_geometric_format(filter_adjacency_list, filter_adjacency_freq_list)
        _, time_edge_weight = list_to_geometric_format(filter_adjacency_list, filter_adjacency_time_list)
        return [edge_index, freq_edge_weight, time_edge_weight]


class ItemGraphBuilder4(GraphRecBuild):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        t = args.t  # 物品图的范围
        limit = args.item_neighbor_num  # 限制物品的数量
        time_span = args.item_graph_span  # 有效时间戳范围

        # 对时间戳列表进行个性化的normalized处理
        scaled_time_list, max_time, min_time = stamp_to_int(timestamp_list)

        # 获取边，不同用户中重复出现的边则span相加
        adjacency_list = []                      # 邻接形式存储物品对
        interval_weight_list = []                # 时间权重
        for i in range(node_num):
            adjacency_list.append(dict())        # dict() 函数用于创建一个字典
            interval_weight_list.append(dict())
        for i in range(len(item_list)):
            tmp_list = item_list[i]
            tmp_time_list = scaled_time_list[i]
            for cur in range(len(tmp_list)):
                u = tmp_list[cur]
                u_time = tmp_time_list[cur]
                for idx in range(max(cur - t, 0), min(cur + t + 1, len(tmp_list))): # t: 物品图建立的窗口
                    v = tmp_list[idx]
                    v_time = tmp_time_list[idx]
                    if u == v:  # 避免self-loop
                        continue
                    if v in adjacency_list[u]:
                        adjacency_list[u][v] = adjacency_list[u][v] + 1
                        interval_weight_list[u][v] = adjacency_list[u][v] + abs(u_time - v_time)
                    else:
                        adjacency_list[u][v] = 1
                        interval_weight_list[u][v] = abs(u_time - v_time)

        filter_adjacency_list = []
        filter_adjacency_freq_list = []
        filter_adjacency_time_list = []
        for i in range(node_num):
            filter_adjacency_list.append([])
            filter_adjacency_freq_list.append([])
            filter_adjacency_time_list.append([])

        # 通过时间间隔进行过滤
        edge_num_before = 0
        edge_num_after = 0
        two_freq_edge_num = 0

        for i in range(node_num):
            cur_item_list = list(adjacency_list[i].items()) # adjacency_list[i].items() 函数以列表返回可遍历的(键,值)
            edge_num_before += len(cur_item_list)
            cur_item_list = sorted(cur_item_list, reverse=True, key=lambda item: item[1])  # 按照频率排序
            for j in range(min(limit, len(cur_item_list))):  # 先限制邻居节点数量，然后在按照时间间隔选取
                v = cur_item_list[j][0]
                freq = cur_item_list[j][1]
                mean_time_span = interval_weight_list[i][v] / freq
                if mean_time_span == 0:          # 两物品间隔太近，视为极大联系
                    mean_time_span = 1
                elif mean_time_span > time_span: # 两物品时间权重超过有效时间戳限制，视为无联系
                    mean_time_span = 0
                if mean_time_span != 0:  # 通过平均时间间隔进行过滤（超过阈值的不放入）
                    filter_adjacency_list[i].append(v)  # 邻居节点id
                    filter_adjacency_freq_list[i].append(freq)  # 边的权重(出现频率)
                    if freq >= 2:
                        two_freq_edge_num = two_freq_edge_num + 1
                    filter_adjacency_time_list[i].append(time_span - mean_time_span)
            edge_num_after += len(filter_adjacency_list[i])

        print("Total edge number %d,edge number after filtering %d with limit %d and time span %d" %
              (edge_num_before, edge_num_after, limit, time_span))
        print("Directed Edge number >= 2 with node limit and time span: %d" % two_freq_edge_num)
        edge_index, freq_edge_weight = list_to_geometric_format(filter_adjacency_list, filter_adjacency_freq_list)
        _, time_edge_weight = list_to_geometric_format(filter_adjacency_list, filter_adjacency_time_list)
        return [edge_index, freq_edge_weight, time_edge_weight]


class UserGraphBuilder1(GraphRecBuild):
    def build_graph(self, node_num, item_list, timestamp_list, args):
        user_sim_array = calc_user_sim(item_list, node_num, args)
        limit = args.item_neighbor_num
        adjacency_list = []
        for i in range(node_num):
            adjacency_list.append([])  # 使用集合去除重复的邻居节点
        # ind = np.argpartition(user_sim_array, -limit)[:, -limit:]  # 对每个用户获取limit个最相似的用户索引
        # print(ind.shape)
        for u in range(node_num):
            if u in user_sim_array:
                cur_list = list(user_sim_array[u].items())
                cur_list = sorted(cur_list, reverse=True, key=lambda user: user[1])
                for i in range(min(len(cur_list), limit)):
                    adjacency_list[u].append(cur_list[i][0])

        edge_index, _ = list_to_geometric_format(adjacency_list)
        return [edge_index]


class MyGraph(object):
    def __init__(self, builder):
        self.builder = builder

    def make_graph(self, node_num, item_list, timestamp_list, cur_args):
        return self.builder.build_graph(node_num, item_list, timestamp_list, cur_args)


def preprocess_dataset(dataset_list, timestamp_list, para_dic, args, item_shift=1):
    print(para_dic)
    # 数据padding,物品id = 0用于padding(原始数据的物品id是从0开始的)
    if item_shift != 0:
        for item_list in dataset_list:
            for i in range(len(item_list)):
                item_list[i] += 1
    # step1: 用户的购买序列list
    user_seq = dataset_list
    split_train_list = []
    for user_id, item_list in enumerate(dataset_list):
        split_train_list.append(item_list[0:-2])  # 每个用户序列分别保留两个物品分别作为 验证集 和 测试集
    assert para_dic['userNum'] == len(user_seq)

    item_node_num = para_dic['itemNum'] + item_shift
    item_info_list = MyGraph(ItemGraphBuilder4()).make_graph(item_node_num, dataset_list, timestamp_list, args)
    print(item_info_list[0].shape)
    print(item_info_list[1].shape)
    print(item_info_list[2].shape)

    user_node_num = para_dic['userNum']
    user_info_list = MyGraph(UserGraphBuilder1()).make_graph(user_node_num, dataset_list, timestamp_list, args)
    print(user_info_list[0].shape)


# ======================   Utils functions for Graph Builder(The following functions) =================================


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


def list_to_matrix_with_weight(row_num, col_num, adjacency_list, weight_adjacency_list):
    row = []
    col = []
    data = []
    for u, neighbor in enumerate(adjacency_list):
        for idx, v in enumerate(neighbor):
            row.append(u)
            col.append(v)
            data.append(weight_adjacency_list[u][idx])  # 边的权重
    # 注意重复的(u,v,data)会造成data的累加
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(row_num, col_num))
    return adjacency_matrix, data


# 将csr格式的矩阵转换为(2,edge_num)的numpy数组,torch geometric需要这个特征
def csr_to_edge_index(csr_mat):
    coo_mat = csr_mat.tocoo(copy=True)
    row_index = coo_mat.row
    col_index = coo_mat.col
    edge_index = np.empty(shape=(2, len(row_index)), dtype=int)
    edge_index[0, :] = row_index
    edge_index[1, :] = col_index
    return edge_index


def list_to_geometric_format(adjacency_list, weight_adjacency_list=None):
    row = []
    col = []
    data = []
    for u in range(len(adjacency_list)):
        neighbor = adjacency_list[u]
        if weight_adjacency_list is not None:
            weight_list = weight_adjacency_list[u]
        for idx, v in enumerate(neighbor):
            row.append(u)
            col.append(v)
            if weight_adjacency_list is not None:
                data.append(weight_list[idx])  # 边的权重
            else:
                data.append(1)

    row_index = np.array(row)
    col_index = np.array(col)
    data_weight = np.array(data)

    edge_index = np.empty(shape=(2, len(row_index)), dtype=int)
    edge_index[0, :] = row_index
    edge_index[1, :] = col_index
    # (2,edge_num),(edge_num)
    return edge_index, data_weight


def stamp_to_int(time_list):
    # 建立相对时间间隔字典
    time_min = int(1e12)  # must be python 3
    for tmp in time_list:
        for tt in tmp:
            time_min = min(time_min, tt)

    time_max = -1
    relative_time_seq = []
    for t_list in time_list:
        time_list = [int(round(float(t - time_min))) for t in t_list]
        time_diff = set()
        # 计算相邻物品之间的时间差列表,time_scale = max(1,min(时间差列表))
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        user_t_min = min(time_list)
        # 时间嵌入: [（购买时间-用户最小购买时间）/ time_scale] + 1 时间嵌入是从1开始的
        cur_list = [int(round((t - user_t_min) / time_scale) + 1) for t in time_list]
        relative_time_seq.append(cur_list)
        time_max = max(time_max, max(cur_list))
    print("数据集中最小时间戳:%d,最大的相邻物品的时间间隔:%d" % (time_min, time_max))
    return relative_time_seq, time_min, time_max


def calc_user_sim(item_list, node_num, args):
    # 构建“物品-用户”倒排索引
    item_user = {}
    for user, items in enumerate(item_list):
        for item in items:
            if item not in item_user:
                item_user[item] = set()
            item_user[item].add(user)

    # user_sim_matrix = np.zeros(shape=(node_num, node_num), dtype=float)
    print('Build user co-rated item matrix ...')
    user_sim_matrix = {}
    for item, users in item_user.items():
        if len(users) > args.hot_item_limit:                # 对热点物品进行限制
            continue
        for u in users:
            for v in users:
                if u == v:
                    continue
                # user_sim_matrix[u][v] = user_sim_matrix[u][v] + 1
                user_sim_matrix.setdefault(u, {})
                user_sim_matrix[u].setdefault(v, 0)
                user_sim_matrix[u][v] += 1
    print("user graph have edge: %d/%d" % (len(user_sim_matrix), node_num))
    # 计算u,v的相似性  = u,v共有的物品数目  / （u的交互的物品 × v交互过的物品总数）
    for u, related_users in user_sim_matrix.items():
        for v, count in related_users.items():
            user_sim_matrix[u][v] = count / math.sqrt(len(item_list[u]) * len(item_list[v]))
    print("finish build !")
    # for u in range(node_num):
    #     for v in range(node_num):
    #         user_sim_matrix[u][v] = user_sim_matrix[u][v] / math.sqrt(len(item_list[u]) * len(item_list[v]))
    return user_sim_matrix


def calc_user_sim_sparse_version(item_list, node_num):
    # 构建“物品-用户”倒排索引
    item_user = {}
    for user, items in enumerate(item_list):
        for item in items:
            if item not in item_user:
                item_user[item] = set()
            item_user[item].add(user)

    user_list = []
    for item, users in item_user.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                user_list.append([u, v])
    print(len(user_list))


if __name__ == "__main__":
    data_path_list = [
        '../DatasetNew/Amazon/preprocess_ratings_Digital_Music.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Beauty.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Books.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_CDs_and_Vinyl.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Movies_and_TV.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Video_Games.pzip',
        '../DatasetNew/MovieLen/preprocess_ml-10m-ratings.pzip',
        '../DatasetNew/MovieLen/preprocess_ml-20m-ratings.pzip'
    ]
    path = data_path_list[0]
    para_dic, sorted_item_list, sorted_time_list = load_decompressed_pzip(path)
    args = parser.parse_args()
    preprocess_dataset(sorted_item_list, sorted_time_list, para_dic, args, item_shift=1)
'''
策略1:
(2, 136568)
(136568,)    

策略2：
{'userNum': 8086, 'itemNum': 15935}
Total edge number 136568, edge number after filtering 81376 with neighbor number limit 10
(2, 81376)
(81376,)

    
策略3:
{'userNum': 8086, 'itemNum': 15935}
数据集中最小时间戳:893721600,最大的相邻物品的时间间隔:5632
Total edge number 136568,edge number after filtering 128792 with  time span 128
Directed Edge number >= 2 without time span  7666, with time span: 7544
(2, 128792)
(128792,)
(128792,)

策略4: 先限制邻居节点数量，然后时间过滤
{'userNum': 8086, 'itemNum': 15935}
数据集中最小时间戳:893721600,最大的相邻物品的时间间隔:5632
Total edge number 136568,edge number after filtering 77623 with limit 10 and time span 128
Directed Edge number >= 2 with node limit and time span: 6670
(2, 77623)
(77623,)
(77623,)

策略5：先时间过滤，然后邻居节点数量过滤
Build user co-rated item matrix ...
Edge similarity matrix construct (8086,8086)
8086
Edge of user graph: 79986
(2, 79986)


'''
