import dgl
import numpy as np
import pandas as pd
import toad
import torch
import torch.nn.functional as F
from dgl.dataloading import NeighborSampler, Sampler
from dgl import set_node_lazy_features, set_edge_lazy_features
from dgl.nn.pytorch import GINConv
from lightning import LightningDataModule
from rtdl_num_embeddings import compute_bins, _check_bins, _PiecewiseLinearEncodingImpl
from sklearn.model_selection import train_test_split

# mycode， 代码根目录写入source，然后就可以同一使用基于该目录的绝对路径
import sys
from pathlib import Path
from typing import List
from torch import nn, Tensor

DIR_SOURCE = str(Path(__file__).resolve().parent.parent)  # 代码根目录写入source，然后就可以同一使用基于该目录的绝对路径
sys.path.append(DIR_SOURCE)
from utils.util import index_to_mask, masks_to_indexs
from ENV import DIR_FRAUD_DATASET

def generate_grid_data(n, m, b):
    # 定义x1和x2的区间
    x_intervals = np.linspace(0, 1, b+1)  #
    y_intervals = np.linspace(0, 1, b+1)  #

    # 所有可能的区间组合
    all_intervals = [(x_intervals[i], x_intervals[i+1], y_intervals[j], y_intervals[j+1])
                     for i in range(b) for j in range(b)]

    # 随机选择m个区间
    chosen_intervals = np.random.choice(range(b*b), m, replace=False)

    samples = []
    for _ in range(n):
        interval_idx = np.random.choice(range(b*b))
        lower_x, upper_x, lower_y, upper_y = all_intervals[interval_idx]

        # 在该区间内生成x1和x2
        x1 = np.random.uniform(lower_x, upper_x)
        x2 = np.random.uniform(lower_y, upper_y)

        if interval_idx in chosen_intervals:
            samples.append((x1, x2, 1))
        else:
            samples.append((x1, x2, 0))

    # 构造DataFrame
    df = pd.DataFrame(samples, columns=['x1', 'x2', 'y'])
    return df


def describe(graph):
    # feat_str = 'feat'
    # trn_msk_str = 'trn_msk'
    # val_msk_str = 'val_msk'
    # tst_msk_str = 'tst_msk'
    # label_str = 'label'
    feat_str = 'feature'
    trn_msk_str = 'train_mask'
    val_msk_str = 'val_mask'
    tst_msk_str = 'test_mask'
    label_str = 'label'

    labeled_mask = graph.ndata[trn_msk_str] | graph.ndata[val_msk_str] | graph.ndata[tst_msk_str]
    # 计算各种类别的数量
    cnum_dict = {}
    if len(graph.ndata[label_str].shape) == 1:
        unique_labels = torch.unique(graph.ndata[label_str])
        for i in unique_labels:
            cnum_dict[i.item()] = torch.sum(graph.ndata[label_str][labeled_mask] == i).item()
    else:
        for i in range(graph.ndata[label_str].shape[1]):
            cnum_dict[i] = torch.sum(graph.ndata[label_str][:, i][labeled_mask]).item()

    nn = graph.number_of_nodes()
    ne = graph.edges()[0].size(0)
    nc = graph.ndata[label_str].max().item() + 1  # Assuming labels are 0-indexed
    d = graph.ndata[feat_str].shape[1]
    trn_size = graph.ndata[trn_msk_str].sum().item()
    val_size = graph.ndata[val_msk_str].sum().item()
    tst_size = graph.ndata[tst_msk_str].sum().item()
    nln = trn_size + val_size + tst_size
    lnr = nln / nn
    lr = [f"{cnum_dict[i] / nln:.2%}" for i in range(nc)]

    print("-" * 80)
    print(f"num nodes {nn:,} | num node feats {d:,} | num edges {ne:,} ")
    print(f"num labeled: {nln:,} | num labeled ratio: {lnr:.2%} | num class: {nc} | class ratio: {', '.join(lr)}")
    print(f"trn size {trn_size:,} | "
          f"val size {val_size:,} | "
          f"tst size {tst_size:,} ")
    print(f"trn rate {trn_size / nln:.2%} | "
          f"val rate {val_size / nln:.2%} | "
          f"tst rate {tst_size / nln:.2%} ")

    # 检查自环
    nodes = graph.nodes()
    self_loops = graph.has_edges_between(nodes, nodes)
    print("有自环的节点的占比：", sum(self_loops).item() / nn)

    # 统计每种边类型的信息
    for etype in graph.etypes:
        subgraph = graph.edge_type_subgraph([etype])
        num_edges = subgraph.number_of_edges()
        avg_out_degree = np.mean(subgraph.out_degrees().numpy())
        # 输出统计信息
        print(f"边{etype}关系下的统计信息:")
        print(f"num edges: {num_edges} | avg out degree: {avg_out_degree:.2f}")
    print("-" * 80)


def read_dataset(data_name):
    print(f'Reading dataset {data_name}...')
    if data_name in ['cora','citeseer', 'pubmed', 'film', 'deezer-europe', 'squirrel', 'chameleon']:
        if data_name in ['cora', 'citeseer', 'pubmed']:
            no_feat_norm_ = True
        else:
            no_feat_norm_ = False
        class args:
            dataset = data_name
            data_dir = SGFormer_DATAPATH
            no_feat_norm = no_feat_norm_
        dataset = load_nc_dataset(args)

        if data_name in ['cora', 'citeseer', 'pubmed']:
            split_idx_lst = [class_rand_splits(dataset.label, 20, 500, 1000)]
        elif data_name in ['deezer-europe']:
            split_idx_lst = [dataset.get_idx_split(train_prop=.5, valid_prop=.25) for _ in range(10)]
        elif data_name in ['film', 'squirrel', 'chameleon']:
            split_idx_lst = load_fixed_splits(dataset, data_name, "semi")
        else:
            print(data_name)
            raise ValueError("Invalid data_name")
        trn_idx = split_idx_lst[0]["train"]
        val_idx = split_idx_lst[0]["valid"]
        tst_idx = split_idx_lst[0]["test"]
        #print(trn_idx)

        num_nodes = dataset.graph['node_feat'].shape[0]
        src_nodes = dataset.graph['edge_index'][0]
        dst_nodes = dataset.graph['edge_index'][1]
        graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        graph = dgl.to_bidirected(graph)
        graph.create_formats_()
        graph.ndata['feature'] = torch.FloatTensor(dataset.graph['node_feat'])
        graph.ndata['label'] = torch.LongTensor(dataset.label)
        graph.ndata['train_mask'] = index_to_mask(trn_idx, num_nodes)
        graph.ndata['val_mask'] = index_to_mask(val_idx, num_nodes)
        graph.ndata['test_mask'] = index_to_mask(tst_idx, num_nodes)
    elif data_name == 'grid_data':
        # 设置参数
        b = 16    # 坐标轴区间数量
        n = 5000  # 样本数量
        m = 25    # 区间数量

        # 生成样本
        df = generate_grid_data(n, m, b)
        num_nodes = len(df)
        graph = dgl.graph((torch.LongTensor([1]), torch.LongTensor([2])), num_nodes=num_nodes)
        graph.ndata['feature'] = torch.FloatTensor(df.values[:,:-1]).contiguous()
        graph.ndata['label'] = torch.LongTensor(df.values[:,-1]).contiguous()
        indices = list(range(len(df)))
        train_indices, test_indices = train_test_split(indices, test_size=0.2)
        graph.ndata['train_mask'] = index_to_mask(train_indices,num_nodes).bool().contiguous()
        graph.ndata['val_mask'] = index_to_mask(test_indices,num_nodes).bool().contiguous()
        graph.ndata['test_mask'] = index_to_mask(test_indices,num_nodes).bool().contiguous()
    else:
        file_path = f'{DIR_FRAUD_DATASET}/{data_name}'
        graph = dgl.load_graphs(file_path)[0][0]
        graph = graph.astype(torch.int64)
        if data_name == 'elliptic':
            graph.ndata['feature'] = graph.ndata['feature'][:, 1:94]
        graph.ndata['train_mask'] = graph.ndata['train_masks'][:, 0].bool().contiguous()
        graph.ndata['val_mask'] = graph.ndata['val_masks'][:, 0].bool().contiguous()
        graph.ndata['test_mask'] = graph.ndata['test_masks'][:, 0].bool().contiguous()
        graph.ndata['feature'] = graph.ndata['feature'].contiguous()
        graph.ndata['label'] = graph.ndata['label'].contiguous()
    describe(graph)
    print('Read dataset done!')
    return graph


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


def bin_one_hot_encoding(x, bins):
    bins = [bin_edges[1:-1] for bin_edges in bins]  # 去掉上下界
    # 初始化一个与x相同形状的tensor，用于存储分箱序号
    bin_indices = torch.zeros_like(x, dtype=torch.long)

    # 遍历每个特征
    for i, edges in enumerate(bins):
        # 对于每个特征，使用torch.bucketize获取分箱序号
        # 注意，torch.bucketize假定bins是有序的
        bin_indices[:, i] = torch.bucketize(x[:, i], edges, right=True)

    x = F.one_hot(bin_indices, num_classes=-1)

    max_n_bins = max(len(edges) for edges in bins) + 1  # 找到最大的分箱数
    mask = torch.row_stack([
        torch.cat(
            [
                torch.ones(len(edges) + 1, dtype=torch.bool, device=x.device),
                torch.zeros(
                    max_n_bins, dtype=torch.bool, device=x.device
                ),
            ]
        )[:max_n_bins]
        for edges in bins
    ])

    x = x[:, mask]
    return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def forward(self, x: Tensor) -> Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """

    def __init__(self, bins: List[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def forward(self, x: Tensor, type='encoding') -> Tensor:
        x = self.impl(x)
        if type == 'encoding':
            return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]
        elif type == 'embedding':
            x = torch.nan_to_num(x,1)
            return x#.view(-1,x.size()[-2]*x.size()[-1])


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        print(h_final)
        return h_final


def preprocess_feats(g, preprocess, n_bins, norm_type=None, gin_n_layers=2):
    trn_idx = g.ndata['train_mask'].bool().nonzero().squeeze()
    feat = g.ndata['feature']
    if norm_type =='None' or norm_type == 'none' or norm_type is None:
        x = feat
    elif norm_type =='std':
        x = (feat - feat.mean(0)) / feat.std(0)
    elif norm_type == 'norm01':
        x = (feat - feat.min(0).values) / (feat.max(0).values - feat.min(0).values)
    elif norm_type == 'gin_norm01':
        print(feat.shape)
        gin = GIN_noparam()
        feat = gin(g)
        print(feat.shape)
        x = (feat - feat.min(0).values) / (feat.max(0).values - feat.min(0).values)
    y = g.ndata['label']
    if preprocess == 'OHE':
        bins = compute_bins(
            x[trn_idx],
            n_bins=n_bins,
            tree_kwargs={'min_samples_leaf': int(float(len(y[trn_idx])) * 0.005) + 1},
            y=y[trn_idx],
            regression=False,
        )
        x = bin_one_hot_encoding(x, bins)
        x = torch.cat([feat, x.float()], dim=-1).contiguous()
    elif preprocess == 'PLEC':
        bins = compute_bins(
            x[trn_idx],
            n_bins=n_bins,
            tree_kwargs={'min_samples_leaf': int(float(len(y[trn_idx])) * 0.005) + 1},
            y=y[trn_idx],
            regression=False,
        )
        # print(bins)
        x = PiecewiseLinearEncoding(bins)(x).contiguous()
    elif preprocess == 'PLEM':
        bins = compute_bins(
            x[trn_idx],
            n_bins=n_bins,
            tree_kwargs={'min_samples_leaf': int(float(len(y[trn_idx])) * 0.005) + 1},
            y=y[trn_idx],
            regression=False,
        )
        # print(bins)
        x = PiecewiseLinearEncoding(bins)(x, type='embedding').contiguous()
    g.ndata['feature'] = x
    print(f"The shape of bin-encoded feature is {x.shape}")
    print('Processing dataset done!\n')


class NodeSampler(Sampler):
    def __init__(
            self,
            prefetch_ndata=None,
            prefetch_edata=None,
            output_device="cpu",
    ):
        super().__init__()
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device

    def sample(self, g, indices):
        """Sampling function

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor
            Placeholder not used.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        node_ids = indices
        sg = g.subgraph(
            node_ids, relabel_nodes=True, output_device=self.output_device
        )
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg


# 构建 neighbor sampling fanout  mini-batch  dataloader DataModule
class MiniFDataModule(LightningDataModule):
    def __init__(self, data_name, **kwargs):
        super().__init__()
        graph = read_dataset(data_name)
        n_bins = kwargs.get('n_bins', 0)
        preprocess = kwargs.get('preprocess', 'None')
        norm_type = kwargs.get('norm_type')
        preprocess_feats(graph, preprocess, n_bins, norm_type)  # 预处理特征，包括归一化、分箱等，默认进行归一化
        self.g = graph
        self.y = graph.ndata['label'].clone().detach()
        self.feat = graph.ndata['feature'].clone().detach()

        if kwargs['device'] == 'cpu':
            self.device = 'cpu'
            self.use_uva = False
        elif kwargs['device'] == 'cuda':
            self.device = 'cuda'
            self.use_uva = True

        self.trn_idx, self.val_idx, self.tst_idx = masks_to_indexs(graph)

        if kwargs['bs']>0 and kwargs['bs']<1:
            self.bs = np.maximum(int(kwargs['bs'] * len(self.trn_idx)),1)
        elif kwargs['bs']==-1:
            self.bs = len(self.trn_idx)
        else:
            self.bs = kwargs['bs']

        if kwargs['val_bs']>0 and kwargs['val_bs']<1:
            self.val_bs = np.maximum(int(kwargs['val_bs'] * graph.number_of_nodes()),1)
        elif kwargs['val_bs']==-1:
            self.val_bs = graph.number_of_nodes()
        else:
            self.val_bs = kwargs['val_bs']

        if 'fanouts' not in kwargs:
            fanouts = [-1] * kwargs.get('gnn_n_layers', 1)
        else:
            fanouts = kwargs['fanouts']
        self.trn_sampler = NeighborSampler(fanouts, prefetch_node_feats=['feature'], prefetch_labels=['label'])
        self.val_sampler = NeighborSampler(fanouts, prefetch_node_feats=['feature'], prefetch_labels=['label'])
        self.d_in = self.g.ndata['feature'].shape[1]
        self.n_classes = self.g.ndata['label'].numpy().max() + 1

    def train_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g, self.trn_idx, self.trn_sampler, device=self.device,
            use_uva=self.use_uva, batch_size=self.bs, shuffle=True, drop_last=True,
        )
        return loader

    def val_dataloader(self):
        bs = self.val_bs
        loader = dgl.dataloading.DataLoader(
            self.g, torch.arange(self.g.num_nodes()), self.val_sampler, device=self.device,
            use_uva=self.use_uva, batch_size=bs, shuffle=True, drop_last=False
        )
        return loader

# 构建 vanilla node sampling subgraph mini-batch dataloader
class MiniGDataModule(LightningDataModule):
    def __init__(self, data_name, **kwargs):
        super().__init__()
        graph = read_dataset(data_name)
        self.n_bins = kwargs.get('n_bins', 0)
        self.preprocess = kwargs.get('preprocess', 'None')
        self.norm_type = kwargs.get('norm_type', 'std')
        preprocess_feats(graph, self.preprocess, self.n_bins, self.norm_type)  # 预处理特征，包括归一化、分箱等，默认进行归一化
        self.g = graph
        self.y = graph.ndata['label'].clone().detach()
        self.feat = graph.ndata['feature'].clone().detach()
        self.bs = kwargs['bs']
        self.val_bs = kwargs['val_bs']
        use_val_fanout = kwargs.get('use_val_fanout', False)
        if kwargs['device'] == 'cpu':
            self.device = 'cpu'
            self.use_uva = False
        elif kwargs['device'] == 'cuda':
            self.device = 'cuda'
            self.use_uva = True

        self.trn_idx, self.val_idx, self.tst_idx = masks_to_indexs(graph)

        if kwargs['bs']>0 and kwargs['bs']<1:
            self.bs = np.maximum(int(kwargs['bs'] * len(self.trn_idx)),1)
        elif kwargs['bs']==-1:
            self.bs = len(self.trn_idx)
        else:
            self.bs = kwargs['bs']

        if kwargs['val_bs']>0 and kwargs['val_bs']<1:
            self.val_bs = np.maximum(int(kwargs['val_bs'] * graph.number_of_nodes()),1)
        elif kwargs['val_bs']==-1:
            self.val_bs = graph.number_of_nodes()
        else:
            self.val_bs = kwargs['val_bs']

        self.d_in = self.g.ndata['feature'].shape[1]
        self.n_classes = self.g.ndata['label'].numpy().max() + 1
        self.drop_last = True

        self.trn_sampler = NodeSampler()
        if use_val_fanout:
            self.val_bs = min(50000, self.g.num_nodes())
            fanout = [-1] * kwargs.get('n_layers', 1)
            self.val_sampler = NeighborSampler(fanout, prefetch_node_feats=['feature'], prefetch_labels=['label'])
            self.shuffle = False
        else:
            if kwargs['val_bs']<1 and kwargs['val_bs']>0:
                self.val_bs = np.maximum(int(kwargs['val_bs'] * graph.number_of_nodes()) , 1)
            elif kwargs['val_bs']==-1:
                self.val_bs = graph.number_of_nodes()
            self.val_sampler = NodeSampler()
            self.shuffle = True


    def train_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g, torch.arange(self.g.num_nodes()), self.trn_sampler, device=self.device,
            use_uva=self.use_uva, batch_size=self.bs, shuffle=True, drop_last=self.drop_last,
        )
        return loader

    def val_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g, torch.arange(self.g.num_nodes()), self.val_sampler, device=self.device,
            use_uva=self.use_uva, batch_size=self.val_bs, shuffle=self.shuffle, drop_last=False
        )
        return loader

if __name__ == '__main__':
    graph = read_dataset('amazon')