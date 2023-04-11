import torch
import torch_geometric
from torch_geometric.data import DataLoader
from util.myutil import load_decompressed_pzip
import random
from time import time

'''
x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
        edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
        ---Coordinate matrix（COO)使用3元组存储,坐标以及该坐标对应的值.
        edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
        y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
        pos (Tensor, optional) – Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
        normal (Tensor, optional) – Normal vector matrix with shape [num_nodes, num_dimensions]. (default: None)
        face (LongTensor, optional) – Face adjacency matrix with shape [3, num_faces]. (default: None)

Batch可以使用Data类的所有方法
     With torch_geometric.data.Data being the base class, all its methods can also be used here. 
     In addition, single graphs can be reconstructed via the assignment vector batch, 
     which maps each node to its respective graph identifier.
Batch对象里面的属性:
<class 'torch_geometric.data.batch.Batch'>
1. num_graphs:查看batch里面有多少个子图.
2. batch: 是一个列向量,torch.Size([所有子图的节点总数]),长度等于这个batch中所有子图的节点总数,
[0,0,0,1,1,1,1,2,2] 比如这个向量表示这个batch有3个子图,第一个子图3个节点,第二个子图4个节点,第三个子图2个节点.
作用: maps each node to its respective graph in the batch(将每一个节点映射到子图中)
3. x: 节点特征   torch.Size([node_num_sum, feat_dim])
4. edge_index: torch.Size([2, edge_num_sum])
5. y: torch.Size([graph_num])
'''


class PairData(torch_geometric.data.Data):
    def __init__(self, edge_index_s, feat_s, bond_feat_s, edge_index_t, feat_t, bond_feat_t, y):
        """

        :param edge_index_s: (2,edge_num)
        :param feat_s:       (node_num,9)
        :param bond_feat_s:  (edge_num,3)
        :param edge_index_t:
        :param feat_t:
        :param bond_feat_t:
        :param class_label: (1)
        """
        super(PairData, self).__init__(y=y)
        # Graph1
        self.edge_index_s = edge_index_s
        self.feat_s = feat_s
        self.bond_feat_s = bond_feat_s
        # Graph2
        self.edge_index_t = edge_index_t
        self.feat_t = feat_t
        self.bond_feat_t = bond_feat_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.feat_s.size(0)
        if key == 'edge_index_t':
            return self.feat_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


def random_divide(num, d_list):
    all_ids = [i for i in range(num)]
    test_num = int(num * d_list[2])  # 1770554　　 测试数据
    valid_num = int(num * d_list[1])
    train_num = num - test_num - valid_num
    print(train_num, valid_num, test_num)
    test_index = random.sample(all_ids, test_num)
    remain_ids = list(set(all_ids) - set(test_index))
    valid_index = random.sample(remain_ids, valid_num)
    train_index = list(set(remain_ids) - set(valid_index))
    assert num == test_num + train_num + valid_num
    assert num == len(train_index) + len(valid_index) + len(test_index)
    # print(len(set.intersection(set(valid_index), set(test_index))))
    # print(len(set.intersection(set(train_index), set(test_index))))
    # print(len(set.intersection(set(train_index), set(valid_index))))
    return train_index, valid_index, test_index


def get_each_dataset(index_list, dataset):
    graph_number_list, node_feat_list, edges_index_list, bond_feat_list = dataset[0]
    datasets_list = []
    for idx in index_list:
        record = dataset[1][idx]
        tmp_id1 = record[0]
        tmp_id2 = record[1]
        label = record[2]
        datasets_list.append(PairData(torch.LongTensor(edges_index_list[tmp_id1]),
                                      torch.LongTensor(node_feat_list[tmp_id1]),
                                      torch.LongTensor(bond_feat_list[tmp_id1]),
                                      torch.LongTensor(edges_index_list[tmp_id2]),
                                      torch.LongTensor(node_feat_list[tmp_id2]),
                                      torch.LongTensor(bond_feat_list[tmp_id2]),
                                      y=label))
    return datasets_list  # each element is pair dataset class


def get_three_dataloader(data_path_list, divide_list, batch_size, is_random=True):
    dataset = [load_decompressed_pzip(file) for file in data_path_list]
    for tmp in dataset:
        print(len(tmp))
    number = len(dataset[1])
    # 01 划分训练集/验证集/测试集
    if is_random:
        t_index, v_index, te_index = random_divide(number, divide_list)
    else:
        print("divide dataset orderly")
        train_number = int(number * divide_list[0])
        test_number = int(number * divide_list[2])
        valid_number = number - train_number - test_number
        t_index = [i for i in range(train_number)]
        v_index = [i for i in range(train_number, train_number + valid_number)]
        te_index = [i for i in range(train_number + valid_number, number)]
    index_lists = [t_index, v_index, te_index]

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
    # new_loader = DataLoader(graph_data_list, batch_size=128, follow_batch=['feat_s', 'feat_t'])
    # new_batch = next(iter(new_loader))
    # print(new_batch.feat_s_batch)           # 这2个batch用于索引邻接矩阵中的边
    # print(new_batch.feat_t_batch)           # 这2个batch用于索引邻接矩阵中的边
    # print(new_batch.feat_s.shape)
    # print(new_batch.feat_t.shape)
    # print(new_batch.edge_index_s.shape)
    # print(new_batch.edge_index_t.shape)
    # print(new_batch.bond_feat_s.shape)
    # print(new_batch.bond_feat_t.shape)
    # print(new_batch.num_graphs)
    '''
        torch.Size([51, 9])
        torch.Size([48, 9])
        torch.Size([2, 102])
        torch.Size([2, 92])
        torch.Size([102, 3])
        torch.Size([92, 3])
    '''

    # 将每一对交互记录转化为一个PairData(torch_geometric.data.Data)类存储起来,说实话感觉这个占用显存比较大

    print("train/valid/test: %d/%d/%d" % (len(t_index), len(v_index), len(te_index)))
    final_lists = []
    for i in range(3):
        final_lists.append(get_each_dataset(index_lists[i], dataset))

    '''
        12733,1819,3638
    '''
    loader_list = []
    shuffle_list = [False, False, False]
    for i in range(3):
        print(len(final_lists[i]))
        loader_list.append(DataLoader(dataset=final_lists[i], batch_size=batch_size, follow_batch=['feat_s', 'feat_t'],
                                      shuffle=shuffle_list[i], num_workers=8, pin_memory=True))
    return loader_list[0], loader_list[1], loader_list[2]  # train/valid/test dataloader


if __name__ == '__main__':
    '''
          edge_attr: [num_edges, num_edge_features]
          edge_index: [2, num_edges]
          torch_geometric.data.Batch inherits from torch_geometric.data.Data 
          and contains an additional attribute called batch
          Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple
           isolated subgraphs), and node and target features are simply concatenated in the node dimension, i.e.
          将每个图的邻接矩阵以对角线的方式拼接成一个大的邻接矩阵,即一个大图中有很多个孤立的子图.
    '''
    print("test the pair dataset!")
    # edge_index_s = torch.tensor([
    #     [0, 0, 0, 0],
    #     [1, 2, 3, 4],
    # ])
    # x_s = torch.randn(5, 16)  # 5 nodes,each node have feature dimension 15
    # edge_index_t = torch.tensor([
    #     [0, 0, 0],
    #     [1, 2, 3],
    # ])
    # x_t = torch.randn(4, 16)  # 4 nodes,each node have feature dimension 4
    # data = PairData(edge_index_s, x_s, edge_index_t, x_t)
    # data_list = [data, data]
    # loader = DataLoader(data_list, batch_size=2)
    # batch = next(iter(loader))
    # print(batch)
    # # 可以用.访问,也可以有['名称']访问
    # # print(batch.x_s)
    # # print(batch.x_t)
    # # print(batch.edge_index_s)
    # # print(batch.edge_index_t)
    # # print(batch['edge_index_s'])
    # # print(batch['x_s'])
    # # print(batch['edge_index_t'])
    # # print(batch['x_t'])
    # new_loader = DataLoader(data_list, batch_size=2, follow_batch=['x_s', 'x_t'])
    # # new_batch = next(iter(new_loader))
    # # print(new_batch)
    # # print(new_batch.x_s_batch)
    # # print(new_batch.x_t_batch)
    # for batch in loader:
    #     print(batch)
    start = time()
    path = "Amazon/"
    suffix_list = [
        "_index_number_feat_edges_attr_list.pzip",  # <list>    # each element is array
        "_interaction_cleaned_indexed.pzip"  # <list>    # each element is [5304, 4981, 1.0]
    ]
    data_id = 0
    dataset_name = ["biosnap_raw"]
    path_lists = []
    for name in dataset_name:
        path_lists.append([path + name + '/' + name + suffix for suffix in suffix_list])
    divide_list = [0.70, 0.10, 0.20]
    a, b, c = get_three_dataloader(path_lists[data_id], divide_list, 2, is_random=False)
    num = 0
    for batch_data in a:
        print(batch_data)
        edge_index_s = batch_data.edge_index_s
        edge_index_t = batch_data.edge_index_t
        feat_s = batch_data.feat_s
        feat_t = batch_data.feat_t
        bond_feat_s = batch_data.bond_feat_s
        bond_feat_t = batch_data.bond_feat_t
        batch_s = batch_data.feat_s_batch
        batch_t = batch_data.feat_t_batch
        label = batch_data.y
        print("-----------")
        print(feat_s.shape)
        print(edge_index_s.shape)
        print(bond_feat_s.shape)
        print(batch_s.shape)
        print("-----------")
        print(feat_t.shape)
        print(edge_index_t.shape)
        print(bond_feat_t.shape)
        print(batch_t.shape)
        print("-----------")
        num += 1
        if num == 2:
            break

