import random
from typing import List

import dgl
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rtdl_num_embeddings import compute_bins, _check_bins, _PiecewiseLinearEncodingImpl
import toad
from torch import Tensor, nn
from tqdm import tqdm
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, )
from sklearn.metrics import confusion_matrix
from sklearn.metrics._ranking import _binary_clf_curve
from lightning.pytorch.callbacks import TQDMProgressBar


# width = 180; print("Terminal width:", width)
# import fcntl, struct, sys, termios; fcntl.ioctl(sys.stdin, termios.TIOCSWINSZ, struct.pack("HHHH", 0, width, 0, 0))


def index_to_mask(index_list, length):
    """
    将给定的索引列表转换为一个长度为length的掩码张量。
    如果输入的是mask张量，也不会有问题，等价于则直接返回。

    参数:
        index_list (list): 包含要转换为掩码的索引的列表。
        length (int): 掩码张量的长度。

    返回:
        mask (torch.Tensor): 一个长度为length的布尔掩码张量，其中给定索引的位置为True，其他位置为False。
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index_list] = True
    return mask


def masks_to_indexs(g):
    trn_idx = g.ndata['train_mask'].nonzero().squeeze()
    val_idx = g.ndata['val_mask'].nonzero().squeeze()
    tst_idx = g.ndata['test_mask'].nonzero().squeeze()
    return trn_idx, val_idx, tst_idx


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 计算获得最优的macrof1,gmean和对应的阈值
def get_max_macrof1_gmean(true, prob):
    fps, tps, thresholds = _binary_clf_curve(true, prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2 * tps / (2 * tps + fns + fps)
    f10 = 2 * tns / (2 * tns + fns + fps)
    marco_f1 = (f11 + f10) / 2

    idx = np.argmax(marco_f1)
    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    gmean = np.sqrt(tps / n_pos * tns / n_neg)
    idx = np.argmax(gmean)
    best_gmean = gmean[idx]
    best_gmean_thr = thresholds[idx]
    return best_marco_f1, best_marco_f1_thr, best_gmean, best_gmean_thr


# 计算所有metrics指标
def cal_binary_metrics(y, prob, trn_idx, val_idx, tst_idx, p='', verbose=False):
    out_dic = {}
    val_th1 = 0
    val_th2 = 0
    for prefix, idx in zip([f'{p}trn_', f'{p}val_', f'{p}tst_'], [trn_idx, val_idx, tst_idx]):
        prob_ = prob[idx]
        y_ = y[idx]
        top_n = sum(y[idx])
        indices = np.argsort(prob_)
        top_n_indices = indices[-top_n:]
        top_n_pred = np.zeros_like(prob_)
        top_n_pred[top_n_indices] = 1

        if prefix in [f'{p}trn_', f'{p}val_']:
            mf1, th1, gme, th2 = get_max_macrof1_gmean(y_, prob_)
            val_th1 = th1
            val_th2 = th2
            pred = np.where(prob_ > th1, 1, 0)
        elif 'tst' in prefix:
            th1 = val_th1
            th2 = val_th2
            pred = np.where(prob_ > th1, 1, 0)
            mf1 = f1_score(y_true=y_, y_pred=pred, average='macro')
            tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
            gme = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

        rec = recall_score(y_, pred)
        pre = precision_score(y_, pred)
        auc = roc_auc_score(y_, prob_)
        aps = average_precision_score(y_, prob_)
        acc = np.mean(np.where(prob_ > 0.5, 1, 0) == y_)
        top = recall_score(y_, top_n_pred)

        dic = {
            f'{prefix}auc': np.round(auc, 5),
            f'{prefix}aps': np.round(aps, 5),  # AP score
            f'{prefix}mf1': np.round(mf1, 5),
            f'{prefix}th1': np.round(th1, 5),
            f'{prefix}gme': np.round(gme, 5),
            f'{prefix}th2': np.round(th2, 5),
            f'{prefix}rec': np.round(rec, 5),
            f'{prefix}pre': np.round(pre, 5),
            f'{prefix}acc': np.round(acc, 5),
            f'{prefix}top': np.round(top, 5),
        }
        formatted_dic = {k: f"{v:.5f}" for k, v in dic.items()}
        if verbose == True:
            print(formatted_dic)
        out_dic.update(dic)
    return out_dic


# 计算所有metrics指标
def cal_multi_metrics(y, pred, trn_idx, val_idx, tst_idx, p='', verbose=False):
    out_dic = {}
    for prefix, idx in zip([f'{p}trn_', f'{p}val_', f'{p}tst_'], [trn_idx, val_idx, tst_idx]):
        pred_ = pred[idx]
        y_ = y[idx]
        acc = np.mean(pred_ == y_)
        dic = {
            f'{prefix}acc': np.round(acc, 5),
        }
        formatted_dic = {k: f"{v:.5f}" for k, v in dic.items()}
        if verbose == True:
            print(formatted_dic)
        out_dic.update(dic)
    return out_dic


# 决策树分箱编码
def bin_encoding(graph, trn_idx, n_bins, col_index=None, verbose=False):
    X = graph.ndata['feature'].numpy()
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    if col_index is None or col_index == 'None':
        col_index = X.columns
    trn_X = X.iloc[trn_idx]
    trn_y = pd.DataFrame(y[trn_idx])
    combiner = toad.transform.Combiner()
    combiner.fit(trn_X, trn_y, method='dt', min_samples=0.01, n_bins=n_bins, )
    # combiner.rules 是一个字典，里面包含了决策树分箱的间断点，打印出不满足分割箱数量的变量名
    if verbose:
        bad_col_list = []
        bad_col_num_list = []
        for col in col_index:
            if len(combiner.rules[col]) < n_bins-1:
                bad_col_list.append(col)
                bad_col_num_list.append(len(combiner.rules[col])+1)
        print(f"The following columns have less than {n_bins} bins: {bad_col_list}")
        print(f"The number of bins for each column: {bad_col_num_list}")
    bins = combiner.export()
    bin_encoded_X = combiner.transform(X[col_index])

    bin_encoded_X_dummies = pd.get_dummies(bin_encoded_X, columns=col_index)
    feature = pd.concat([X, bin_encoded_X_dummies], axis=1)

    feature = feature.astype(float)
    print(f"The shape of bin-encoded feature is {feature.shape}") if verbose is True else None
    return feature




class MyProgressBar(TQDMProgressBar):  # pycharm中val显示不太正常
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


def format_time(seconds):
    # 分别计算小时、分钟和秒
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 根据时间长度决定是否显示对应的时间单位
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0:  # 保证在有小时时即使分钟为0也显示
        parts.append(f"{int(minutes)}min")
    parts.append(f"{int(seconds)}s")

    return " ".join(parts)

# model_paramspace_map['SGFormer'] = {
#     # model params
#     'd_hidden': {"values": [64,70]},
#     'n_layers': {"value": 2},
#     'dropout': {"value": 0.5},
#     'n_heads': {"value": 1},
#     'alpha': {"value": 0.5},  # 这个是res连接历史层的权重超参
#     'graph_weight': {"value": 0.8},  # 这个是融gnn的权重超参
#     'use_residual': {"value": True},
#     'ours_n_layers': {"value": 1},
#     'ours_dropout': {"value": 0.2},
#     'ours_use_residual': {"value": False},
#     'ours_use_act': {"value": False},
#     'use_graph': {"value": True},
#     'use_bn': {"value": False},
#     # training params
#     'lr': {"value": 0.01},
#     'weight_decay': {"value": 0.0005},
#     'ours_weight_decay': {"value": 0.001},
#     # dataloader params
#     'bs': {"value": -1},
#     'no_feat_norm': {"value": True},
#     'seed': {"value": 123},
#     # split params
#     'split_type': {"value": 'rand_split_class'},
#     'valid_num': {"value": 500},
#     'test_num': {"value": 1000},
# }


if __name__ == '__main__':
    # 忽略特定类型的警告
    #warnings.filterwarnings("ignore", category=dgl.base.DGLWarning)
    file_path = '/home/dmj/rhspace/001_GADBench/datasets/yelp'
    graph = dgl.load_graphs(file_path)[0][0]
    graph.ndata['train_mask'] = graph.ndata['train_masks'][:,0].contiguous()
    graph.ndata['val_mask'] = graph.ndata['val_masks'][:,0].contiguous()
    graph.ndata['test_mask'] = graph.ndata['test_masks'][:,0].contiguous()
    graph.ndata['feature'] = graph.ndata['feature'].contiguous()
    graph.ndata['label'] = graph.ndata['label'].contiguous()
    print('Read dataset done!')
    describe(graph)

    print('Fanout Minibatch:')
    print('cpu-cpu-cpu,-1 -1')
    fanout = [-1,-1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=256, shuffle=True, drop_last=True)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        input_nodes = input_nodes.to('cuda')
        output_nodes = output_nodes.to('cuda')
        for block in blocks:
            block = block.to('cuda')
        pass

    print('cpu-cpu-cpu,-1 -1, enable_cpu_affinity4')
    fanout = [-1, -1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=256, shuffle=True, drop_last=True,num_workers=4)
    with loader.enable_cpu_affinity():
        for input_nodes, output_nodes, blocks in tqdm(loader):
            input_nodes = input_nodes.to('cuda')
            output_nodes = output_nodes.to('cuda')
            for block in blocks:
                block = block.to('cuda')
            pass

    print('cpu-cpu-cpu,-1 -1, enable_cpu_affinity16')
    fanout = [-1, -1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=256, shuffle=True, drop_last=True, num_workers=16)
    with loader.enable_cpu_affinity():
        for input_nodes, output_nodes, blocks in tqdm(loader):
            input_nodes = input_nodes.to('cuda')
            output_nodes = output_nodes.to('cuda')
            for block in blocks:
                block = block.to('cuda')
            pass

    print('cpu-cpu-cuda,-1 -1')
    fanout = [-1,-1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout)
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        batch_size=256, shuffle=True, drop_last=True)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        pass

    print('cpu-cpu-cuda,-1 -1, prefetch')
    fanout = [-1,-1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout, prefetch_node_feats=['feature'], prefetch_labels=['label'])
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        batch_size=256, shuffle=True, drop_last=True)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        pass

    print('cpu-cpu-cuda,-1 -1, prefetch, use_uva')
    fanout = [-1, -1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout, prefetch_node_feats=['feature'], prefetch_labels=['label'])
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        use_uva=True, batch_size=256, shuffle=True, drop_last=True, num_workers=0)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        pass

    print('cpu-cpu-cuda,-1 -1, prefetch, use_uva')
    fanout = [-1, -1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout, prefetch_node_feats=['feature'], prefetch_labels=['label'])
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        use_uva=True, batch_size=256, shuffle=True, drop_last=True,
                                        )
    for input_nodes, output_nodes, blocks in tqdm(loader):
        pass


    print('cpu-cpu-cuda,-1 -1, prefetch, num_workers=10')
    fanout = [-1, -1]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout, prefetch_node_feats=['feature'], prefetch_labels=['label'])
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        batch_size=256, shuffle=True, drop_last=True, num_workers=10)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        pass

    print('cpu-cpu-cpu,5 25')
    fanout = [5, 25]
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.NeighborSampler(fanout, prefetch_node_feats=['feature'],prefetch_labels=['label'])
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=512, shuffle=True, drop_last=True)
    for input_nodes, output_nodes, blocks in tqdm(loader):
        input_nodes = input_nodes.to('cuda')
        output_nodes = output_nodes.to('cuda')
        for block in blocks:
            block = block.to('cuda')
        pass

    print('NodeFormer Minibatch:')
    print('cpu-cpu-cpu')
    graph = graph.to('cpu')
    trn_sampler = NodeSampler(output_device='cpu')
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=512, shuffle=True, drop_last=True)
    for subgraph in tqdm(loader):
        subgraph = subgraph.to('cuda')
        pass
    subgraph = dgl.remove_self_loop(subgraph)
    describe(subgraph.to('cpu'))

    print('cpu-cpu-cuda')
    graph = graph.to('cpu')
    trn_sampler = NodeSampler(output_device='cpu')
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        batch_size=512, shuffle=True, drop_last=True)
    for subgraph in tqdm(loader):
        pass
    subgraph = dgl.remove_self_loop(subgraph)
    describe(subgraph.to('cpu'))

    print('cpu-cpu-cpu')
    graph = graph.to('cpu')
    trn_sampler = NodeSampler(output_device='cpu')
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cpu",
                                        batch_size=graph.num_nodes(), shuffle=True, drop_last=True)
    # with loader.enable_cpu_affinity():
    for subgraph in tqdm(loader):
        subgraph = subgraph.to('cuda')
        pass
    subgraph = dgl.remove_self_loop(subgraph)
    describe(subgraph.to('cpu'))

    print('SAINTRW Minibatch:')
    file_path = '/home/dmj/rhspace/001_GADBench/datasets/yelp'
    graph = dgl.load_graphs(file_path)[0][0]
    graph.ndata['train_mask'] = graph.ndata['train_masks'][:, 0].contiguous()
    graph.ndata['val_mask'] = graph.ndata['val_masks'][:, 0].contiguous()
    graph.ndata['test_mask'] = graph.ndata['test_masks'][:, 0].contiguous()
    graph.ndata['feature'] = graph.ndata['feature'].contiguous()
    graph.ndata['label'] = graph.ndata['label'].contiguous()
    print('Read dataset done!')
    graph = graph.to('cpu')
    trn_sampler = dgl.dataloading.SAINTSampler(mode='walk', budget=[50, 200],output_device='cpu')
    loader = dgl.dataloading.DataLoader(graph, torch.arange(graph.num_nodes()), trn_sampler, device="cuda",
                                        batch_size=10000, shuffle=True, drop_last=False)
    for subgraph in tqdm(loader):
        pass

    subgraph = dgl.remove_self_loop(subgraph)
    describe(subgraph.to('cpu'))

