import copy
import torch
import torch.nn as nn
import argparse
from util.myutil import *
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from DatasetNew.Mymodel1Dataset_v4 import get_loader_graph_info_rating_matrix, get_loader_graph_info_rating_matrix_v1
from tqdm import tqdm
'''
1）重新实现一下HGN系列模型的训练集，测试集和验证的训练框架‘
2）采用多线程技术提高模型预测的效率，ndgc和HR实现按照batch计算，然后合并
-----------------------------------------
a）基于自注意力机制的模型:
    1）单个全局图信息提取模块i
    2）SasRec模块
    3）用户嵌入表示
    4）通道注意力机制
    
b) 基于HAM的模型



'''


# =================================p1:全局图信息提取的参考代码======================================================
# torch geometric 提供的GCN层


class GlobalGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GlobalGCNLayer, self).__init__()
        # 官方库中函数默认进行normalized以及加上self-loop,此外加上了bias,注意官方函数库并没有考虑edge attributes
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=output_dim, add_self_loops=False)

    def forward(self, feats, edges, weight=None):
        """
        :param weight:
        :param feats: (node_num,feat_dim)
        :param edges: (2,edge_num)
        :return: (node_num,output_dim)
        """
        if weight is None:
            x = self.conv1(feats, edges)
        else:
            x = self.conv1(feats, edges, weight)
        return x  # (node_num,output_dim)


class GlobalInfoGenerate(nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gcn'):
        """
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        """
        super(GlobalInfoGenerate, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.dropout_layer = nn.Dropout(p=drop_ratio)

        # add residual connection or not()
        self.residual = residual

        # List of GNNs: the number of GNN layer = self.num_layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'gcn':
                self.convs.append(GlobalGCNLayer(input_dim=emb_dim, output_dim=emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, input_embedding_layer, edges_index, weight=None):
        # input_embedding_layer:GNN的网络输入层
        # [原始输入,第一层输出,第二层输出,第三层输出]
        h_list = [input_embedding_layer.weight]
        for layer in range(self.num_layer):
            if weight is None:
                h = self.convs[layer](h_list[layer], edges_index)
            else:
                h = self.convs[layer](h_list[layer], edges_index, weight)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)  # 可考虑删除非线性激活层F.relu()
            h = self.dropout_layer(h)
            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        return node_representation


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# Intermediate用于构建单层transform
class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# 单层transform
class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        # Layer 就是单层transform
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# ===========================p4:模型主体代码===============================================================================

# 对物品图进行GCN的嵌入表示 过时版本，未使用
class DGRecV0(nn.Module):
    def __init__(self, args):
        super(DGRecV0, self).__init__()
        # 物品和位置的嵌入
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # 基于自注意力机制的编码器
        self.item_encoder = Encoder(args)

        # layerNorm层定义和dropout层定义
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # 物品图的特征提取
        self.item_graph_encoder = GlobalInfoGenerate(num_layer=args.item_GNN_layer_num,
                                                     emb_dim=args.hidden_size,
                                                     drop_ratio=args.item_GNN_dropout,
                                                     JK="sum",
                                                     residual=False,
                                                     gnn_type='gcn')

        if args.device == 'cpu':
            self.cuda_condition = False
        else:
            self.cuda_condition = True

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)  # 参数初始化

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transformer_encoder(self, sequence):
        # s1:物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # s3:mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # s4:对序列进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def mixed_encoder(self, sequence, item_edge_index):
        # 物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        # 物品的图的特征提取
        # input_embedding_layer, edges_index
        item_g_output = self.item_graph_encoder(input_embedding_layer=self.item_embeddings,
                                                edges_index=item_edge_index)
        item_rel_info = F.embedding(sequence, item_g_output)  # 从物品GNN输出索引

        # 加上位置向量
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # 加上物品图的关系特征
        sequence_emb = sequence_emb + item_rel_info

        # mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 采用自注意力机制对用户序列本身进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    # 模型验证时，计算当前序列的表示×所有物品的嵌入 得到 该用户对每个物品的兴趣
    def infer(self, sequence):
        # 获取编码后序列中最后一个的嵌入表示（短期兴趣）
        out = self.transformer_encoder(sequence)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(out, test_item_emb.transpose(0, 1))
        return rating_pred

    def mixed_infer(self, sequence, item_edge_index):
        # 获取编码后序列中最后一个的嵌入表示（短期兴趣）
        out = self.mixed_encoder(sequence, item_edge_index)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(out, test_item_emb.transpose(0, 1))
        return rating_pred


# ===================================================================
# 考虑了用户图和物品图
class DGRecV1(nn.Module):
    def __init__(self, args):
        super(DGRecV1, self).__init__()
        # 物品和位置的嵌入
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # 用户嵌入表示
        self.user_embedding = nn.Embedding(args.user_size, args.hidden_size)

        # 基于自注意力机制的编码器
        self.item_encoder = Encoder(args)

        # layerNorm层定义和dropout层定义
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

        # 物品图的特征提取
        if args.item_rel:
            self.item_graph_encoder = GlobalInfoGenerate(num_layer=args.item_GNN_layer_num,
                                                         emb_dim=args.hidden_size,
                                                         drop_ratio=args.item_GNN_dropout,
                                                         JK="last",
                                                         residual=False,
                                                         gnn_type='gcn')

        if args.user_rel:
            self.user_graph_encoder = GlobalInfoGenerate(num_layer=args.user_GNN_layer_num,
                                                         emb_dim=args.hidden_size,
                                                         drop_ratio=args.user_GNN_dropout,
                                                         JK="last",
                                                         residual=False,
                                                         gnn_type='gcn')

        if args.device == 'cpu':
            self.cuda_condition = False
        else:
            self.cuda_condition = True

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)  # 参数初始化

    # 参数初始化
    def init_weights(self, module):
        """ Initialize the weights.
        """
        # isinstance() 函数来判断一个对象是否是一个已知的类型
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # 用于CoDGRec
    def transformer_encoder(self, sequence):
        # s1:物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # s3:mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # s4:对序列进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    # 用于CoDGRec
    def rel_transformer_encoder(self, sequence, item_edge_index, item_edge_weight=None):
        # s1:物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        item_g_output = self.item_graph_encoder(input_embedding_layer=self.item_embeddings,
                                                edges_index=item_edge_index, weight=item_edge_weight)
        item_rel_info = F.embedding(sequence, item_g_output)  # 从物品GNN输出索引

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        # 将原始物品嵌入信息修改为物品关联信息  !!!!!!!!!!!!!!!!!!!!!!!!!!
        sequence_emb = item_rel_info + position_embeddings + item_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # s3:mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # s4:对序列进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def mixed_encoder(self, sequence, item_edge_index, batch_user, user_edge_index, item_edge_weight=None):
        # 物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        # 物品的图的特征提取
        # input_embedding_layer, edges_index
        if self.args.item_rel:
            item_g_output = self.item_graph_encoder(input_embedding_layer=self.item_embeddings,
                                                    edges_index=item_edge_index, weight=item_edge_weight)
            item_rel_info = F.embedding(sequence, item_g_output)  # 从物品GNN输出索引

        # 用户图的特征聚合
        orgin_user_emb = self.user_embedding(batch_user)
        if self.args.user_rel:
            user_g_output = self.user_graph_encoder(input_embedding_layer=self.user_embedding,
                                                    edges_index=user_edge_index)
            '''
            import torch.nn.functional as F,以下为与 torch.nn 的不同，来源 https://blog.csdn.net/wangweiwells/article/details/100531264
                            torch.nn.X	                            torch.nn.functional.X
                                是类	                                        是函数
                      结构中包含所需要初始化的参数	             需要在函数外定义并初始化相应参数,并作为参数传入
            一般情况下放在_init_ 中实例化,并在forward中完成操作	一般在_init_ 中初始化相应参数,在forward中传入
            '''
            '''
            input参数是我们想要用向量表示的对象的索引，weight储存了我们的向量表示，这个函数的目的就是输出一个索引和向量的对应关系。
            来源：https://blog.csdn.net/rouge_eradiction/article/details/124288799
            '''
            user_emb = F.embedding(batch_user, user_g_output)  # (batch,hidden) 用户聚会后的表示

        # 加上位置向量
        # 函数torch.arange()返回大小的一维张量
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # unsqueeze()的作用是用来增加给定tensor的维度的，unsqueeze(dim)就是在维度序号为dim的地方给tensor增加一维
        # 将张量扩展为参数tensor的大小
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # 改动1：加上物品图的关系特征
        if self.args.item_rel:
            sequence_emb = sequence_emb + item_rel_info
        else:
            sequence_emb = sequence_emb

        # mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 采用自注意力机制对用户序列本身进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        transform_output = item_encoded_layers[-1]

        # 改动2: 加上用户的嵌入表示（原始嵌入+关联信息）
        if self.args.user_rel:
            user_rel_info = (user_emb + orgin_user_emb).unsqueeze(1).repeat(1, seq_length, 1)  # 用户嵌入表示沿水平方向扩展
        else:
            user_rel_info = orgin_user_emb.unsqueeze(1).repeat(1, seq_length, 1)  # 用户嵌入表示沿水平方向扩展

        output = transform_output + user_rel_info
        return output

    # 用于evaluate_model
    def mixed_infer(self, sequence, item_edge_index, batch_user, user_edge_index, item_edge_weight=None):
        # 获取编码后序列中最后一个的嵌入表示（短期兴趣）
        out = self.mixed_encoder(sequence, item_edge_index, batch_user, user_edge_index, item_edge_weight)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(out, test_item_emb.transpose(0, 1))
        return rating_pred


# -----------------------------------------------------------------------------------------------------
# 考虑用户图和物品图，加入通道注意力机制和用户隐藏兴趣单元
class DGRecV2(nn.Module):
    def __init__(self, args):
        super(DGRecV2, self).__init__()
        # 物品和位置的嵌入
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # 用户嵌入表示
        self.user_embedding = nn.Embedding(args.user_size, args.hidden_size)
        # 用户记忆网络
        self.user_memory_unit = UserMemoryUnit(args.hidden_size, args.unit_num)

        # 基于自注意力机制的编码器
        self.item_encoder = Encoder(args)

        # layerNorm层定义和dropout层定义
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # 物品图的特征提取
        self.item_graph_encoder = GlobalInfoGenerate(num_layer=args.item_GNN_layer_num,
                                                     emb_dim=args.hidden_size,
                                                     drop_ratio=args.item_GNN_dropout,
                                                     JK="last",
                                                     residual=False,
                                                     gnn_type='gcn')
        # 用户图特征提取
        self.user_graph_encoder = GlobalInfoGenerate(num_layer=args.user_GNN_layer_num,
                                                     emb_dim=args.hidden_size,
                                                     drop_ratio=args.user_GNN_dropout,
                                                     JK="last",
                                                     residual=False,
                                                     gnn_type='gcn')
        # 基于注意力机制的特征融合
        # self.info_fuse = InfoFuse(hidden_size=args.hidden_size, drop_ratio=args.fuse_dropout_ratio)

        if args.device == 'cpu':
            self.cuda_condition = False
        else:
            self.cuda_condition = True

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)  # 参数初始化

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transformer_encoder(self, sequence):
        # s1:物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # s3:mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # s4:对序列进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def mixed_encoder(self, sequence, item_edge_index, batch_user, user_edge_index):
        # 物品序列的嵌入
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)

        # 物品的图的特征提取
        # input_embedding_layer, edges_index
        item_g_output = self.item_graph_encoder(input_embedding_layer=self.item_embeddings,
                                                edges_index=item_edge_index)
        item_rel_info = F.embedding(sequence, item_g_output)  # 从物品GNN输出索引

        # 用户图的特征聚合
        user_g_output = self.user_graph_encoder(input_embedding_layer=self.user_embedding,
                                                edges_index=user_edge_index)
        user_emb = F.embedding(batch_user, user_g_output)  # (batch,hidden) 用户聚会后的表示
        origin_user_emb = self.user_embedding(batch_user)

        # 加上位置向量
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # 加上物品图的关系特征
        sequence_emb = sequence_emb + item_rel_info

        # mask weight的计算
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 采用自注意力机制对用户序列本身进行编码
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        transform_output = item_encoded_layers[-1]

        # 通过记忆网络学习用户长期潜在兴趣推理
        user_latent_interest = self.user_memory_unit(transform_output, user_emb)
        '''
                user_rel_info = user_emb.unsqueeze(1).repeat(1, seq_length, 1)  # 用户嵌入表示沿水平方向扩展
                output = transform_output + user_rel_info
                没有考虑记忆网络
        '''
        output = transform_output + user_latent_interest
        # 通过注意力机制获取综合信息表示
        # output = self.info_fuse(origin_user_emb=origin_user_emb,
        #                         seq_out=transform_output,
        #                         item_rel=item_rel_info,
        #                         memory_interest=user_latent_interest)
        return output

    def mixed_infer(self, sequence, item_edge_index, batch_user, user_edge_index):
        # 获取编码后序列中最后一个的嵌入表示（短期兴趣）
        out = self.mixed_encoder(sequence, item_edge_index, batch_user, user_edge_index)[:, -1, :]
        test_item_emb = self.item_embeddings.weight
        rating_pred = torch.matmul(out, test_item_emb.transpose(0, 1))
        return rating_pred


# 训练阶段没法整
class UserMemoryUnit(nn.Module):
    def __init__(self, hidden_size, unit_num):
        super(UserMemoryUnit, self).__init__()
        key_data = torch.zeros(hidden_size, unit_num).type(torch.FloatTensor)
        value_data = torch.zeros(hidden_size, unit_num).type(torch.FloatTensor)
        self.memory_keys = nn.parameter.Parameter(data=key_data, requires_grad=True)
        self.memory_values = nn.parameter.Parameter(data=value_data, requires_grad=True)
        self.init_params()

    def forward(self, seq_output, user_embbing):
        """

        :param seq_output: [batch,max_len,hidden]
        :param user_rel:   [batch,hidden]
        :return:
        """
        # 展开seq_out
        (batch, max_len, hidden) = seq_output.shape
        user_rel_info = user_embbing.unsqueeze(1).repeat(1, max_len, 1)  # (batch,max_len,hidden)
        interest_query = user_rel_info + seq_output  # 这里相加是否合适?
        query = interest_query.reshape(-1, hidden)  # (batch*max_len,hidden)，amount = batch*max——len

        # (amount,hidden) * (hidden,m) = (amount,m)                     # 每个物品获得m个记忆单元的注意力得分
        query_score = torch.matmul(query, self.memory_values)
        softmax_query_score = F.softmax(query_score, dim=1)

        out = torch.matmul(softmax_query_score, self.memory_values.T)  # (amount,m) * (hidden,m)^T
        user_latent_interest = out.reshape(batch, max_len, hidden)
        return user_latent_interest

    def init_params(self):
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)


class InfoFuse(nn.Module):
    def __init__(self, hidden_size, drop_ratio):
        super(InfoFuse, self).__init__()

        # 3 layer MLP calculate attention score
        self.att1 = nn.Linear(hidden_size * 2, hidden_size)
        self.att2 = nn.Linear(hidden_size, hidden_size)
        self.att3 = nn.Linear(hidden_size, 1)
        self.dropout_layer = nn.Dropout(p=drop_ratio)
        self.init_params()

    def forward(self, origin_user_emb, seq_out, item_rel, memory_interest):  # (batch,item_num,hidden)
        (batch, max_len, hidden) = seq_out.shape
        origin_embeddings = origin_user_emb.unsqueeze(1).repeat(1, 3, 1)  # (batch,3,hidden)
        embed = origin_embeddings.unsqueeze(1).repeat(1, max_len, 1, 1)  # (batch,item_num,3,hidden)
        other = torch.stack((seq_out, item_rel, memory_interest), 2)  # (batch,item_num,3,hidden)

        x = torch.cat((embed, other), 3)  # (batch,item_num,3,2*hidden)
        x = F.relu(self.att1(x))  # (batch,item_num,3,hidden)
        x = self.dropout_layer(x)
        # x = F.relu(self.att2(x))
        # x = self.dropout_layer(x)
        x = self.att3(x)
        att = F.softmax(x, dim=0)  # (batch,item_num,3,1)

        b_score = att.permute(0, 1, 3, 2).reshape(-1, 1, 3)  # (batch,item_num,1,3) => (batch*item_num,1,3)
        b_other = other.reshape(-1, 3, hidden)  # (batch*item_num,3,hidden)
        final_out = torch.bmm(b_score, b_other).squeeze()  # (batch*item_num,1,hidden)

        return final_out.reshape(batch, max_len, hidden)  # (batch,item_num,hidden)

    def init_params(self):
        nn.init.xavier_uniform_(self.att1.weight)  # glorot_uniform initialization,for Relu function
        nn.init.constant_(self.att1.bias, 0)
        nn.init.xavier_uniform_(self.att2.weight)  # glorot_uniform initialization,for Relu function
        nn.init.constant_(self.att2.bias, 0)


# ------------------------------------------------------------------------------------------------------
# 未使用
class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embeddings = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def update_embedding_matrix(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)
        self.base_embedding_matrix = self.item_embeddings(self.total_item_list)

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(item_idx).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (self.max_score - self.min_score)
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list

# 未使用
class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


if __name__ == '__main__':
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
    parser.add_argument("--hot_item_limit", default=100, type=int)  # 用户建图协同过滤算法的共现矩阵不考虑超热点物品（提高效率）
    # number of user memory unit
    parser.add_argument("--unit_num", default=10, type=int)  # 用户长期兴趣的记忆单元数量
    # parameters of fuse module
    parser.add_argument("--fuse_dropout_ratio", default=0.5, type=float)
    args = parser.parse_args()

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

    data_loader_list, item_g_info, user_g_info, rating_matrix = get_loader_graph_info_rating_matrix_v1(path,
                                                                                                       batch_size=128,
                                                                                                       max_seq_len=50,
                                                                                                       args=args)
    (useNum, itemSize) = rating_matrix[0].shape  # 这里物品的数目 = 实际物品数目 + 1 （用于padding）
    args.item_size = itemSize
    args.user_size = useNum
    args.cuda_condition = True
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    args.cuda_condition = True  # 是否使用GPU
    test_model = DGRecV2(args).to(args.device)  #
    print("模型参数 %d" % model_parameters_calculation(test_model))

    # item_edge_index = torch.LongTensor(item_g_info[0]).to(args.device)  # 获取item graph (2,edge_num) torch.tensor
    # user_edge_index = torch.LongTensor(user_g_info[0]).to(args.device)
    #
    # train_loader = data_loader_list[0]
    # rec_cf_data_iter = enumerate(train_loader)
    # for i, rec_batch in rec_cf_data_iter:
    #     rec_batch = tuple(t.to(args.device) for t in rec_batch)
    #     batch_user, input_ids, target_pos, target_neg, _ = rec_batch
    #     sequence_output = test_model.mixed_encoder(input_ids, item_edge_index, batch_user, user_edge_index)
    #     print(sequence_output.shape)
    #     break

    # item_seq = torch.randint(1, 10, (10, 5))  # 64个用户，输入的物品序列长度为5
    # user_ids = torch.randint(1, 10, (10,))  # 64个用户id
    # items_to_predict = torch.randint(1, 10, (10, 3 + 3))  # 训练模式需要负样本(T=3个为正样本，3个为负样本)
    # if args.cuda_condition:
    #     item_seq = item_seq.cuda()
    #     user_ids = user_ids.cuda()
    #     items_to_predict = items_to_predict.cuda()
    # '''
    # 输入:
    #     item_seq:(batch,物品序列id长度)
    #     user_ids:(batch，)
    #     items_to_predict:(batch,预测物品的id长度）
    #     -训练:id长度=正样本个数+负样本个数
    #     -预测:id长度=所有物品id个数(需要对所有物品评分)
    # 输出: (batch,预测物品id的长度)
    # '''
    # test_out = test_model.transformer_encoder(item_seq)
    # print(test_out.shape)
    #
    # '''
    #    另外一种思路:
    #    1)建立全局图去捕获用户的长期兴趣
    #    2)transform捕获的是用户短期兴趣
    #    3)二个维度的数据增强
    #    设计思路上:transform的encoder的训练方法与HGN训练方法不一样，HGN需要自己手动的切分L+T的训练样本。
    # '''

    '''
    torch.Size([50, 10])
    torch.Size([50, 10])
    <class 'torch.nn.parameter.Parameter'>
    <class 'torch.nn.parameter.Parameter'>
    '''
    memory_module = UserMemoryUnit(hidden_size=50, unit_num=10)
    # item_seq = torch.randint(1, 10, (10, 5))  # 64个用户，输入的物品序列长度为5
    # user_ids = torch.randint(1, 10, (10,))    # 64个用户id
    # print(memory_module.memory_keys.shape)
    # print(memory_module.memory_values.shape)
    # print(type(memory_module.memory_keys))
    # print(type(memory_module.memory_values))
