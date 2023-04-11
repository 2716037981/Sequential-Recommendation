from models.MyModel1 import DGRecV1
import argparse
import logging
import datetime
import torch
import numpy as np
import scipy.sparse as sp
from util.myutil import *
from DatasetNew.Mymodel1Dataset_v4 import get_loader_graph_info_rating_matrix_v1
import time

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--dir', type=int, default=3)

# train arguments
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--seed", default=999, type=int)

# learning related
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

# model dependent arguments
parser.add_argument("--model_name", default='DGSasRec_V1', type=str)
parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument('--num_attention_heads', default=2, type=int)
parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument('--max_seq_length', default=50, type=int)
# - global item graph parameters
parser.add_argument('--item_GNN_layer_num', default=2, type=int)
parser.add_argument('--item_GNN_dropout', default=0.5, type=float)
# - global user graph parameters
parser.add_argument('--user_GNN_layer_num', default=2, type=int)
parser.add_argument('--user_GNN_dropout', default=0.5, type=float)
# - item/user graph build hyperparameters
parser.add_argument('--t', default=2, type=int)  # 窗口的大小
parser.add_argument('--item_graph_span', default=128, type=int)  # 时间戳嵌入的范围
parser.add_argument("--item_neighbor_num", default=5, type=int)  # 限制每个物品的邻居节点数量
parser.add_argument("--user_neighbor_num", default=5, type=int)  # 限制每个用户的邻居节点数量
parser.add_argument("--hot_item_limit", default=100, type=int)  # 用户建图协同过滤算法的共现矩阵不考虑超热点物品（提高效率）

# 探究权重对于模型的影响
parser.add_argument('--info_use', default=2, type=int, help="0:use item frequency,1:use time interval,2:use both ")
# 探究关联信息对于模型的影响
parser.add_argument('--user_rel', default=True, type=bool, help="whether use information from user relationship")
parser.add_argument('--item_rel', default=False, type=bool, help="whether use information from item relationship")

# 模型训练阶段的loss计算
def train_cross_entropy_calc(args, model, seq_out, pos_ids, neg_ids):
    # 模型训练阶段的loss计算
    # [batch seq_len hidden_size]
    pos_emb = model.item_embeddings(pos_ids)
    neg_emb = model.item_embeddings(neg_ids)
    # [batch*seq_len hidden_size]
    pos = pos_emb.view(-1, pos_emb.size(2))
    neg = neg_emb.view(-1, neg_emb.size(2))
    seq_emb = seq_out.view(-1, args.hidden_size)  # [batch*seq_len hidden_size]
    pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
    neg_logits = torch.sum(neg * seq_emb, -1)
    istarget = (pos_ids > 0).view(pos_ids.size(0) * args.max_seq_length).float()  # [batch*seq_len]
    loss = torch.sum(
        - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
        torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    ) / torch.sum(istarget)
    return loss

# _, valid_res = evaluate_model(args, model, data_loader_list[1], rating_matrix[0], item_edge_index, user_edge_index)
def evaluate_model(args, model_, dataset_loader, cur_rating_matrix, item_edge_index, user_edge_index, weight=None):
    rec_data_iter = enumerate(dataset_loader)
    model_.eval()
    pred_list = None
    answer_list = None
    for i, batch in rec_data_iter:
        batch = tuple(t.to(args.device) for t in batch)
        user_ids, input_ids, target_pos, target_neg, answers = batch
        rating_pred = model_.mixed_infer(input_ids, item_edge_index, user_ids, user_edge_index, weight)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        # 根据模型预测的得分进行排序获取top20的物品
        batch_user_index = user_ids.cpu().numpy()
        rating_pred[cur_rating_matrix[batch_user_index].toarray() > 0] = 0
        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        # argpartition T: O(n)  argsort O(nlogn)
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if i == 0:
            pred_list = batch_pred_list
            answer_list = answers.cpu().data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

    # hgn_get_full_sort_score(answer_list, pred_list)       # hgn提供的计算函数，计算结果是一样的
    return get_full_sort_score(answer_list, pred_list)


def main(dataset_idx=0):
    start_time = time.time()

    args = parser.parse_args()
    # 根据数据集调整参数!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if dataset_idx in [2, 4]:       # dropout of movielen  should be lower
        args.hidden_dropout_prob = 0.2
        args.attention_probs_dropout_prob = 0.2
        args.max_seq_length = 200

    setup_seed(args.seed)
    '''=================================step2:数据处理，生成模型训练和测试所需要的numpy数据=================================='''
    data_path_list = [
        '../DatasetNew/Amazon/preprocess_ratings_Digital_Music.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Video_Games.pzip',
        '../DatasetNew/MovieLen/preprocess_ml-10m-ratings.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Beauty.pzip',
        '../DatasetNew/MovieLen/preprocess_ml-20m-ratings.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_Movies_and_TV.pzip',
        '../DatasetNew/Amazon/preprocess_ratings_CDs_and_Vinyl.pzip',
        '../DatasetNew/HIN_Dataset/preprocess_my_yelp.pzip',
        '../DatasetNew/HIN_Dataset/preprocess_my_LastFM.pzip',
    ]
    #  '../DatasetNew/Amazon/preprocess_ratings_Books.pzip',   # 该数据集无法使用，物品数目太多，评估浪费时间
    args.dir = dataset_idx
    curPath = data_path_list[args.dir]
    data_loader_list, item_g_info, user_g_info, rating_matrix = get_loader_graph_info_rating_matrix_v1(curPath,
                                                                                                       batch_size=args.batch_size,
                                                                                                       max_seq_len=args.max_seq_length,
                                                                                                       args=args)

    '''====================================step3:模型训练============================================================='''
    (useNum, itemSize) = rating_matrix[0].shape  # 这里物品的数目 = 实际物品数目 + 1 （用于padding）

    args.cuda_condition = True  # 是否使用GPU
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")
    args.train = True

    checkpoint = args.model_name + '.pt'  # 最佳模型保存位置
    args.checkpoint_path = os.path.join("../savedModel/", checkpoint)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)

    args.item_size = itemSize
    args.user_size = useNum
    model = DGRecV1(args).to(args.device)
    betas = (args.adam_beta1, args.adam_beta2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    dataloader = data_loader_list[0]

    if args.info_use == 0:
        item_edge_weight = torch.FloatTensor(item_g_info[2]).to(args.device)  # 获取item graph (2,edge_num) torch.tensor
        print("Edge weight is time interval!")
    elif args.info_use == 1:
        item_edge_weight = torch.FloatTensor(item_g_info[1]).to(args.device)  # 获取item graph (2,edge_num) torch.tensor
        print("Edge weight is item frequency!")
    elif args.info_use == 2:
        item_edge_weight = torch.FloatTensor(item_g_info[1]+item_g_info[2]).to(args.device)
        print("Edge weight is time interval +  item frequency")
    item_edge_index = torch.LongTensor(item_g_info[0]).to(args.device)
    user_edge_index = torch.LongTensor(user_g_info[0]).to(args.device)
    end_epoch = 300
    for epoch in range(args.epochs):
        if args.train is False:
            break
        model.train()
        rec_avg_loss = 0.0
        rec_cf_data_iter = enumerate(dataloader)
        ans = 0
        for i, rec_batch in rec_cf_data_iter:
            rec_batch = tuple(t.to(args.device) for t in rec_batch)
            batch_user, input_ids, target_pos, target_neg, _ = rec_batch
            sequence_output = model.mixed_encoder(input_ids, item_edge_index, batch_user, user_edge_index,
                                                  item_edge_weight)

            rec_loss = train_cross_entropy_calc(args, model, sequence_output, target_pos, target_neg)
            optimizer.zero_grad() # 梯度设置为0
            rec_loss.backward() # 反向传播，计算当前梯度
            optimizer.step() # 根据梯度更新网络参数
            rec_avg_loss += rec_loss.item()
            ans += 1

        # 模型验证，保存最佳模型
        scores, _ = evaluate_model(args, model, data_loader_list[1], rating_matrix[0], item_edge_index, user_edge_index)
        post_fix = {
            "epoch": epoch + 1,
            "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / ans),
            "valid-HIT@10": '{:.4f}'.format(scores[2]),
            "valid-NDCG@10": '{:.4f}'.format(scores[3])
        }
        print(post_fix)
        early_stopping(np.array(scores[-1:]), model)  # 将所有指标合并为1个numpy数组
        if early_stopping.early_stop:
            print("Early stopping")
            end_epoch = epoch
            break

    # 模型在测试集上评估
    print("==========================================================================")
    print("load best model for valid dataset")
    model.load_state_dict(torch.load(args.checkpoint_path))
    _, valid_res = evaluate_model(args, model, data_loader_list[1], rating_matrix[0], item_edge_index, user_edge_index)
    print("result for valid dataset:" + valid_res)
    _, test_res = evaluate_model(args, model, data_loader_list[2], rating_matrix[1], item_edge_index, user_edge_index)
    print("result for test dataset:" + test_res)
    print("=============================================================================")
    # write res in txt
    res_path = "../Result/DGRecV1.txt"
    write_to_txt(file_path=res_path, content_str=curPath)
    write_to_txt(file_path=res_path, content_str="result for valid dataset:" + valid_res)
    write_to_txt(file_path=res_path, content_str="result for test dataset:" + test_res)
    duration = int(time.time() - start_time)
    write_to_txt(file_path=res_path, content_str="Time consume: " + str(duration) + " sec " + str(end_epoch)
                                                 + " epoch stop")
    write_to_txt(file_path=res_path, content_str=str(args))
    write_to_txt(file_path=res_path, content_str="=============================================")


# 模型训练瓶颈在于验证集对所有物品的计算时间
if __name__ == '__main__':
        main(dataset_idx=0)

