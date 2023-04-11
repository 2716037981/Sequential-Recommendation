import argparse
import glob
import os
import pickle
import random
import shutil
import numpy as np
import torch
import h5py
import bz2
import gzip
import _pickle as cPickle
import math


# for large files, compress and decompress consume time
def store_compressed_pbz2(souce_object, path_file):
    with bz2.BZ2File(path_file, 'wb', compresslevel=9) as f:
        cPickle.dump(souce_object, f)


def load_decompressed_pbz2(path_file):
    db_file = bz2.BZ2File(path_file, 'rb')
    data = cPickle.load(db_file)
    db_file.close()
    return data


# for large files, compress and decompress consume time
def store_compressed_pzip(souce_object, path_file):
    with gzip.GzipFile(path_file, 'wb', compresslevel=6) as f:
        cPickle.dump(souce_object, f)


def load_decompressed_pzip(path_file):
    db_file = gzip.GzipFile(path_file, 'rb')
    data = cPickle.load(db_file)
    db_file.close()
    return data


# one-hot编码，没有占位符
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


# 带有占位符的one-hot编码
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def store_hdf5(object_list, path_file, group_list=None, level=3):
    if group_list is None:
        group_list = ['smile1', 'smile2', 'label']
    print(group_list)
    f = h5py.File(path_file, 'w')
    for idx, group in enumerate(group_list):
        f.create_dataset(name=group, data=object_list[idx], compression="gzip", compression_opts=level)
    f.close()


def load_hdf5(path_file, group_list=None):
    if group_list is None:
        group_list = ['smiles1', 'smiles2', 'label']
    f = h5py.File(path_file, 'r')
    value_list = []
    for key in group_list:
        value_list.append(np.array(f.get(key)[:]))
    f.close()
    return value_list


def store_pkl(souce_object, path_file):
    # wb: write file and use binary mode
    dbfile = open(path_file, 'wb')
    # source, destination
    pickle.dump(souce_object, dbfile)
    dbfile.close()


def load_pkl(path_file):
    # for reading also binary mode is important
    dbfile = open(path_file, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db


# def adjacent_matrix(mol):
#     adjacency = Chem.GetAdjacencyMatrix(mol)
#     # return np.array(adjacency) + np.eye(adjacency.shape[0])
#     return np.array(adjacency)  # do not use self-loop


def save_best_model(model, model_dir, best_epoch):
    # save parameters of trained model
    torch.save(model.state_dict(), model_dir + '{}.pkl'.format(best_epoch))
    files = glob.glob(model_dir + '*.pkl')
    # delete models saved before
    for file in files:
        tmp = file.split('/')[-1]  # windows:\\  linux: /
        tmp = tmp.split('.')[0]
        epoch_nb = int(tmp)
        if epoch_nb < best_epoch:
            os.remove(file)


# calculation sum of model parameters
def model_parameters_calculation(model):
    return sum([torch.numel(param) for param in model.parameters()])


def assert_dir_exist(x):
    if not os.path.exists(x):
        os.makedirs(x)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("fix random seed !")


def value_to_hot(value_array):
    label_num = value_array.max() + 1
    num = value_array.shape[0]
    value_list = value_array.tolist()
    one_hot_array = np.zeros(shape=(num, label_num))
    for row, value in enumerate(value_list):
        one_hot_array[row, value] = 1
    return one_hot_array


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_file_name_from_dic(para_dic):
    keys_list = para_dic.keys()
    filename = ''
    for tmp_key in keys_list:
        filename += str(tmp_key) + ':' + str(para_dic[tmp_key]) + '|'
    filename += '.csv'
    return filename


def str_list_to_int_list(str_list):
    str1 = str_list[1: -1].split(',')
    int_list = [int(i) for i in str1]
    return int_list


def change_para_dic_text_cci(parser, para_dic):
    para_dic['lr'] = parser.lr
    para_dic['epochs'] = parser.epochs
    para_dic['dropout'] = parser.dropout
    para_dic['batch'] = parser.batch
    para_dic['pooling_size'] = parser.pooling_size
    para_dic['channel_list'] = str_list_to_int_list(parser.channel_list)
    para_dic['filter_size_smiles'] = str_list_to_int_list(parser.filter_size_smiles)
    para_dic['mlp_sizes'] = str_list_to_int_list(parser.mlp_sizes)


def write_to_txt(file_path, content_str):
    with open(file_path, 'a+') as f:
        f.write(content_str + '\n')


def str_list_of_list_to_int_list(str_lists):
    strs = str_lists[1: -1].split('_')
    final_lists = [str_list_to_int_list(tmp) for tmp in strs]
    return final_lists


def show_model_parameters(model):
    print("--------------parameters start!---------------")
    for i in model.parameters():
        print(i.shape)
    number = sum([torch.numel(param) for param in model.parameters()])
    print("Sum of parameters for model： %d" % number)
    print("--------------parameters end!---------------")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def get_full_sort_score(answers, pred_list):
    recall, ndcg = [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
        "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
        "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
    }
    return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)


if __name__ == "__main__":
    print("test")
    # para_dic = {
    #     'smile_embedding_dim': 32,
    #     'dropout': 0.1,
    #     'channel_list': [32, 64, 128, 256],
    #     'filter_size_smiles': [4, 6, 8],
    #     'batch': 32,
    #     'lr': 1e-3,  # learning rate
    #     'seed': 20,  # random seed
    #     'epochs': 150,
    #     'CUDA': True,
    #     'stop_counter': 200,
    #     'is_train': True,
    #     'is_test': True,
    #     'class_number': 2,
    # }
    # print(para_dic)
    # print(os.altsep)
    # 在Windows系统下的分隔符是：\ (反斜杠)。
    # 在Linux系统下的分隔符是： / （斜杠）。
    # model_dir = '../savedModel/savedTextCCI/' + 'cci900' + '/'
    # best_epoch = 10
    # files = glob.glob(model_dir + '*.pkl')
    # # delete models saved before
    # for file in files:
    #     print(file)
    #     tmp = file.split('\\')[-1]
    #     print(tmp)
    #     tmp = tmp.split('.')[0]
    #     epoch_nb = int(tmp)
    #     if epoch_nb < best_epoch:
    #         os.remove(file)
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.is_available())
    # with open('./data.txt', 'r') as f:
    #     data = f.read()
    #     print('context: {}'.format(data))
    # str_list_of_list_to_int_list('[[3, 1]_[5, 1]]')
    # store_hdf5(None, 'test.hdf5', group_list=None)
    # object_list = load_hdf5('test.hdf5')
    # for tmp in object_list:
    #     print(tmp.shape)
