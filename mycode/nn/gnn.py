# dgl
import dgl.function as fn
import dgl
# 机器学习
# torch
import torch
import torch as th
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from rtdl_num_embeddings import compute_bins
from torch.nn import Identity
from torch.nn.functional import relu, silu, elu, selu, gelu, sigmoid, tanh, softmax, log_softmax, dropout
from dgl import DGLError
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, check_eq_shape
# 工程化、自建和其他
# pl
from pprint import pprint
from torch import nn
from torch.nn.parameter import Parameter

# my code
torch.set_float32_matmul_precision('medium')


class DyPLELayer(nn.Module):
    def __init__(self, d_in, n_bins, dy_raw_bin_width=True, n_heads=1):
        super().__init__()
        self.n_bins = n_bins
        self.dy_raw_bin_width = dy_raw_bin_width
        self.n_heads = n_heads
        self.raw_bin_width = nn.Parameter(torch.randn(d_in, self.n_heads, self.n_bins),
                                          requires_grad=self.dy_raw_bin_width)
        self.register_buffer('mask', torch.tril(torch.ones(self.n_bins, self.n_bins)))
        self.o_d_in = d_in
        mask_left = torch.ones((d_in, self.n_bins))
        mask_left[:, -1] = 0
        self.register_buffer('mask_left', mask_left.bool())
        mask_right = torch.ones((d_in, self.n_bins))
        mask_right[:, 0] = 0
        self.register_buffer('mask_right', mask_right.bool())
        self.d_out = self.o_d_in * self.n_heads * self.n_bins

    def forward(self, inp):
        #bs = inp.shape[0]
        bin_width = self.raw_bin_width.softmax(dim=-1)
        bin_axis = (bin_width[:, :, None, :] * self.mask[None, None, :, :]).sum(dim=-1)
        zeros = torch.zeros((bin_axis.shape[0], self.n_heads, 1), device=bin_width.device)
        new_bin_axis = torch.cat((zeros, bin_axis), dim=-1)[..., :self.n_bins]
        diff = inp[:, :, None, None] - new_bin_axis
        rate = diff / bin_width
        # x = F.relu(1-F.relu(1-rate)) # activation function
        rate = rate.transpose(1, 2).flatten(-2, -1)
        rate[:, :, self.mask_left.flatten()] = 1 - F.relu(1 - rate[:, :, self.mask_left.flatten()])
        rate[:, :, self.mask_right.flatten()] = F.relu(rate[:, :, self.mask_right.flatten()])
        x = rate.view(-1, self.o_d_in, self.n_bins)
        return x

    def init_params(self, x, y):
        n_bins = self.n_bins
        mask_left = torch.ones((self.o_d_in, self.n_bins))
        mask_right = torch.ones((self.o_d_in, self.n_bins))
        bins = compute_bins(
            x,
            n_bins=n_bins,
            tree_kwargs={'min_samples_leaf': int(float(len(y)) * 0.005) + 1},
            y=y,
            regression=False,
        )
        pprint(bins)
        bins_matrix = torch.zeros((x.shape[1], n_bins + 1))
        for i, bin in enumerate(bins):
            l = len(bin)
            bins_matrix[i, -l:] = bin
            mask_left[i, -1] = 0
            mask_right[i, 1 - l] = 0
        logs = (bins_matrix.diff() + 1e-8).log()
        s = -torch.mean(logs)
        raw_bin_width = logs + s
        self.raw_bin_width.data.copy_(raw_bin_width.unsqueeze(1))
        self.mask_left.data.copy_(mask_left)
        self.mask_right.data.copy_(mask_right)
        print("dyple in and out:", self.o_d_in, self.d_out)


class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings."""

    def __init__(self, n: int, in_features: int, out_features: int, bias=True) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.b = bias
        if self.b:
            self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.b:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1):] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        if self.b:
            x = x + self.bias
        return x


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, **kwargs):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = nn.ReLU()

        d_hidden = kwargs['d_hidden']

        preprocess = kwargs['preprocess']
        n_bins = kwargs['n_bins']
        use_dyple = kwargs['use_dyple']

        pre_n_layers = kwargs['pre_n_layers']
        pre_use_bn = kwargs['pre_use_bn']
        pre_dropout = kwargs['pre_dropout']

        self.fc_in = nn.Sequential()

        if not use_dyple and preprocess in ['None', 'PLEC']:
            pass
        elif not use_dyple and preprocess in ['PLEM']:
            self.fc_in.append(_NLinear(d_in, n_bins, n_bins))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            d_in = d_in * kwargs.get('n_bins')
        elif use_dyple and preprocess in ['None']:
            self.dy_ple_layer = DyPLELayer(d_in, kwargs.get('n_bins'), kwargs.get('dy_raw_bin_width'), 1)
            self.fc_in.append(self.dy_ple_layer)
            self.fc_in.append(_NLinear(d_in, kwargs.get('n_bins'), kwargs.get('n_bins')))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            d_in = d_in * kwargs.get('n_bins')
        else:
            raise ValueError('use_dyple', use_dyple, 'preprocess', preprocess)

        if pre_n_layers == 0:
            self.fc_in.append(nn.Identity())
        else:
            for i in range(pre_n_layers):
                if i == 0:
                    self.fc_in.append(nn.Linear(d_in, d_hidden))
                else:
                    self.fc_in.append(nn.Linear(d_hidden, d_hidden))
                self.fc_in.append(nn.ReLU())
                self.fc_in.append(nn.BatchNorm1d(d_hidden)) if pre_use_bn else None
                self.fc_in.append(nn.Dropout(pre_dropout))
            d_in = d_hidden
        self.fc_out = nn.Linear(d_in, n_classes)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_out(x)
        return x


class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=False,
            norm=None,
            activation=None,
    ):
        super().__init__()
        valid_aggre_types = {"mean", "gcn", "max_pool", "mean_pool"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn

        self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            if "pool" in self._aggre_type:
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                if self._aggre_type == "max_pool":
                    graph.update_all(msg_fn, fn.max("m", "neigh"))
                if self._aggre_type == "mean_pool":
                    graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class SSAGEConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = feat
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            h_self = feat_dst
            graph.srcdata["h"] = feat_src
            msg_fn = fn.copy_u("h", "m")
            graph.update_all(msg_fn, fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            return h_neigh


class SAGE(nn.Module):
    def __init__(self, d_in, n_classes=2, **kwargs):
        super().__init__()
        d_hidden = kwargs['d_hidden']


        preprocess = kwargs['preprocess']
        n_bins = kwargs['n_bins']
        use_dyple = kwargs['use_dyple']

        pre_n_layers = kwargs['pre_n_layers']
        pre_use_bn = kwargs['pre_use_bn']
        pre_dropout = kwargs['pre_dropout']

        gnn_n_layers = kwargs['gnn_n_layers']
        gnn_use_bn = kwargs['gnn_use_bn']
        gnn_dropout = kwargs['gnn_dropout']
        gnn_agg = kwargs['gnn_agg']
        self.gnn_use_res = kwargs['gnn_use_res']

        self.use_mha = kwargs['use_mha']
        mha_n_layers = kwargs['mha_n_layers']
        mha_n_heads = kwargs['mha_n_heads']

        if self.use_mha:
            self.mha = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=mha_n_heads, dropout=0., batch_first=True)

        self.act = getattr(nn, 'ReLU')()
        self.fc_in = nn.Sequential()

        # PLE 相关
        if not use_dyple and preprocess in ['None', 'PLEC']:
            pass
        elif not use_dyple and preprocess in ['PLEM']:
            self.fc_in.append(_NLinear(d_in, n_bins, n_bins))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            d_in = d_in * kwargs.get('n_bins')
        elif use_dyple and preprocess in ['None']:
            self.dy_ple_layer = DyPLELayer(d_in, kwargs.get('n_bins'), kwargs.get('dy_raw_bin_width'), 1)
            self.fc_in.append(self.dy_ple_layer)
            self.fc_in.append(_NLinear(d_in, kwargs.get('n_bins'), kwargs.get('n_bins')))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            d_in = d_in * kwargs.get('n_bins')
        else:
            raise ValueError(f'use_dyple: {use_dyple} preprocess: {preprocess}')

        # 前置mlp
        if pre_n_layers == 0:
            self.fc_in.append(nn.Identity())
        else:
            for i in range(pre_n_layers):
                if i == 0:
                    self.fc_in.append(nn.Linear(d_in, d_hidden))
                else:
                    self.fc_in.append(nn.Linear(d_hidden, d_hidden))
                self.fc_in.append(nn.ReLU())
                self.fc_in.append(nn.BatchNorm1d(d_hidden)) if pre_use_bn else None
                self.fc_in.append(nn.Dropout(pre_dropout))
            d_in = d_hidden

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_n_layers):
            self.gnn_layers.append(SAGEConv(d_in, d_hidden, gnn_agg, activation=self.act, feat_drop=gnn_dropout,
                                            norm=nn.BatchNorm1d(d_hidden) if gnn_use_bn else None))
            d_in=d_hidden

        self.fc_out = nn.Linear(d_in, n_classes)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # 遍历模型的所有模块
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, blocks, h):
        h0 = h = self.fc_in(h)
        # v = h[:blocks[-1].num_dst_nodes()]
        # 逐层应用GraphSAGE卷积
        h_in = h
        for i, (block, layer) in enumerate(zip(blocks, self.gnn_layers)):
            h = layer(block, h_in)
            if self.gnn_use_res:
                h = 0.7 * h + 0.3 * h_in[:block.num_dst_nodes()]
            h_in = h

        if self.use_mha:
            v = q = k = h[:blocks[-1].num_dst_nodes()]
            o, _ = self.mha(q, k, v)
            o = self.act(o)
            h = 0.5 * o + 0.5 * h
        # 最后的全连接层
        h = self.fc_out(h)
        return h


class DyPLEEmbedding(nn.Module):
    def __init__(self, d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout, use_feat_emb=False):
        super().__init__()
        self.fc_in = nn.Sequential()
        self.drop = nn.Dropout(pre_dropout)
        self.dy_ple_layer = DyPLELayer(d_in, n_bins, dy_raw_bin_width, 1)
        self.nlinear = _NLinear(d_in, n_bins, d_feat_emb)
        self.bn0 = nn.BatchNorm1d(d_in) if pre_use_bn else nn.Identity()
        self.linear = nn.Linear(d_in * d_feat_emb, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden) if pre_use_bn else nn.Identity()
        self.use_feat_emb = use_feat_emb

    def forward(self, h):
        h = self.dy_ple_layer(h)
        h = self.nlinear(h)
        h = self.bn0(h)
        feat_h = h = F.relu(h, inplace=True)
        h = self.drop(h)
        h = h.flatten(start_dim=1)
        h = self.linear(h)
        h = self.bn1(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)
        if self.use_feat_emb:
            return h, feat_h
        else:
            return h

    def init_params(self, x, y):
        self.dy_ple_layer.init_params(x, y)




class SAGEHis(nn.Module):
    def __init__(self, d_in, n_classes=2, **kwargs):
        super().__init__()
        d_hidden = kwargs['d_hidden']
        agg = kwargs['agg']

        preprocess = kwargs['preprocess']
        n_bins = kwargs['n_bins']
        use_dyple = kwargs['use_dyple']
        dy_raw_bin_width = kwargs['dy_raw_bin_width']

        pre_n_layers = kwargs['pre_n_layers']
        pre_use_bn = kwargs['pre_use_bn']
        pre_dropout = kwargs['pre_dropout']

        gnn_n_layers = kwargs['gnn_n_layers']
        gnn_use_bn = kwargs['gnn_use_bn']
        gnn_dropout = kwargs['gnn_dropout']
        self.gnn_use_res = kwargs['gnn_use_res']

        self.use_mha = kwargs['use_mha']
        mha_n_layers = kwargs['mha_n_layers']
        mha_n_heads = kwargs['mha_n_heads']

        self.use_presage = kwargs['use_presage']

        self.act = getattr(nn, 'ReLU')()
        self.fc_in = nn.Sequential()

        # PLE 相关
        if not use_dyple and preprocess in ['None', 'PLEC']:
            pass
        elif not use_dyple and preprocess in ['PLEM']:
            self.fc_in.append(_NLinear(d_in, n_bins, n_bins))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            self.fc_in.append(nn.Linear(d_in * n_bins, d_hidden))
            d_in = d_hidden
        elif use_dyple and preprocess in ['None']:
            self.dy_ple_embedding = DyPLEEmbedding(d_in, d_hidden, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout)
            self.dy_ple_embeddings = nn.ModuleList([self.dy_ple_embedding])
            if self.use_presage:
                for i in range(gnn_n_layers):
                    self.dy_ple_embeddings.append(
                        DyPLEEmbedding(d_in, d_hidden, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout))
            self.dy_ple_embedding = self.dy_ple_embeddings[0]
            self.fc_in.append(self.dy_ple_embedding)
            d_in = d_hidden
        else:
            raise ValueError(f'use_dyple: {use_dyple} preprocess: {preprocess}')

        # 前置mlp
        if pre_n_layers == 0:
            self.fc_in.append(nn.Identity())
        else:
            for i in range(pre_n_layers):
                if i == 0:
                    self.fc_in.append(nn.Linear(d_in, d_hidden))
                else:
                    self.fc_in.append(nn.Linear(d_hidden, d_hidden))
                self.fc_in.append(nn.ReLU())
                self.fc_in.append(nn.BatchNorm1d(d_hidden)) if pre_use_bn else None
                self.fc_in.append(nn.Dropout(pre_dropout))
            d_in = d_hidden

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_n_layers):
            self.gnn_layers.append(SAGEConv(d_in, d_hidden, agg, activation=self.act, feat_drop=gnn_dropout,
                                            norm=nn.BatchNorm1d(d_hidden) if gnn_use_bn else None))

        if self.use_mha:
            self.mha = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=mha_n_heads, dropout=0., batch_first=True)

        self.fc_out = nn.Linear(d_in, n_classes)

        n_nodes = kwargs['n_nodes']
        self.ssage = SSAGEConv()
        self.register_buffer('his_emb', torch.rand((n_nodes, d_hidden)))

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # 遍历模型的所有模块
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, blocks, h):
        h_list = []
        h_hat = h
        h_tilde = self.fc_in(h_hat)
        h_list.append(h_tilde[:blocks[-1].num_dst_nodes()])
        # 逐层应用GraphSAGE卷积
        gnn_in = h_tilde
        for i, (block, layer) in enumerate(zip(blocks, self.gnn_layers)):
            h = layer(block, gnn_in)
            if self.use_presage:
                h_hat = self.ssage(block, h_hat)
                h_tilde = self.dy_ple_embeddings[i](h_hat)
                h_list.append(h_tilde[:blocks[-1].num_dst_nodes()])
            if self.gnn_use_res:
                h = 0.7 * h + 0.3 * gnn_in[:block.num_dst_nodes()]
            gnn_in = h

        if self.use_presage:
            h = 0.5 * torch.stack(h_list).mean(dim=0) + 0.5 * h

        if self.use_mha:
            v = k = self.his_emb
            q = h[:blocks[-1].num_dst_nodes()]
            o, _ = self.mha(q, k, v)
            o = self.act(o)
            h = 0.1 * o + 0.9 * h

        # 最后的全连接层
        h = self.fc_out(h)
        return h, gnn_in


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
            bias=True,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._in_feats = in_feats
        # if isinstance(in_feats, tuple):
        #     self.fc_src = nn.Linear(
        #         self._in_src_feats, out_feats * num_heads, bias=False
        #     )
        #     self.fc_dst = nn.Linear(
        #         self._in_dst_feats, out_feats * num_heads, bias=False
        #     )
        # else:
        #     self.fc = nn.Linear(
        #         self._in_src_feats, out_feats * num_heads, bias=False
        #     )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, in_feats, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, in_feats, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # if hasattr(self, "fc"):
        #     nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # else:
        #     nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        #     nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, x, h, get_attention=False):
        with (graph.local_scope()):
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            src_prefix_shape = dst_prefix_shape = x.shape[:-1]
            h_src = h_dst = self.feat_drop(x).view(
                *src_prefix_shape, 1, self._in_feats, 1
            )
            feat_src = feat_dst = h
            # h.view(
            #     *src_prefix_shape, self._num_heads, 1, self._out_feats
            # )
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
                h_dst = h_dst[: graph.number_of_dst_nodes()]
                dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": h_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            # graph.update_all(fn.copy_u("ft", "m"), fn.mean("m", "ft"))
            rst = graph.dstdata["ft"]
            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class PrePMP(nn.Module):
    def __init__(self, d_in, n_classes=2, **kwargs):
        super().__init__()
        d_hidden = kwargs['d_hidden']
        agg = kwargs['agg']

        preprocess = kwargs['preprocess']
        n_bins = kwargs['n_bins']
        use_dyple = kwargs['use_dyple']
        dy_raw_bin_width = kwargs['dy_raw_bin_width']
        d_feat_emb = kwargs['d_feat_emb']
        #
        pre_n_layers = kwargs['pre_n_layers']
        pre_use_bn = kwargs['pre_use_bn']
        pre_dropout = kwargs['pre_dropout']

        gnn_n_layers = kwargs['gnn_n_layers']
        gnn_use_bn = kwargs['gnn_use_bn']
        gnn_dropout = kwargs['gnn_dropout']
        self.gnn_use_res = kwargs['gnn_use_res']

        # self.use_mha = kwargs['use_mha']
        # mha_n_layers = kwargs['mha_n_layers']
        # mha_n_heads = kwargs['mha_n_heads']

        self.act = getattr(nn, 'ReLU')()
        self.fc_in = nn.Sequential()

        # # 前置mlp
        # if pre_n_layers == 0:
        #     self.fc_in.append(nn.Identity())
        # else:
        #     for i in range(pre_n_layers):
        #         self.fc_in.append(nn.Linear(d_in, d_hidden))
        #         self.fc_in.append(nn.ReLU())
        #         self.fc_in.append(nn.BatchNorm1d(d_hidden)) if pre_use_bn else None
        #         self.fc_in.append(nn.Dropout(pre_dropout))
        #         d_in = d_hidden
        if not use_dyple and preprocess in ['PLEC', 'None']:
            self.embedding0 = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.BatchNorm1d(d_hidden) if pre_use_bn else nn.Identity(),
                self.act,
                nn.Dropout(pre_dropout),
            )
        elif use_dyple and preprocess == 'None':
            self.embedding0 = DyPLEEmbedding(d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width,
                                             pre_use_bn, pre_dropout, True)

        self.gnn_layers = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        for i in range(gnn_n_layers):
            self.gnn_layers.append(
                GATConv(d_in, d_feat_emb, num_heads=1, bias=False, attn_drop=gnn_dropout)  # , activation=self.act , residual=True
            )
            if not use_dyple and preprocess in ['PLEC', 'None']:
                self.embeddings.append(
                    nn.Sequential(
                        nn.Linear(d_in, d_hidden),
                        self.act,
                        nn.BatchNorm1d(d_hidden) if pre_use_bn else nn.Identity(),
                        nn.Dropout(pre_dropout),
                    )
                )
            elif use_dyple and preprocess == 'None':
                self.embeddings.append(
                    DyPLEEmbedding(d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width,
                                   pre_use_bn, pre_dropout, True)
                )
        self.fc_out = nn.Linear(d_hidden * (gnn_n_layers + 1), n_classes)

    def forward(self, blocks, x):
        num_dst_nodes = blocks[-1].num_dst_nodes()
        h_list = []
        h, feat_h = self.embedding0(x)
        h_list.append(h[:num_dst_nodes])
        for i, (block, embedding, layer) in enumerate(zip(blocks, self.embeddings, self.gnn_layers)):
            x = layer(block, x, feat_h).flatten(1)
            h, feat_h = embedding(x)
            h_list.append(h[:num_dst_nodes])
        h_final = torch.cat(h_list, -1)
        # 最后的全连接层
        h = self.fc_out(h_final)
        return h


class DyPLEFeatEmbedding(nn.Module):
    def __init__(self, d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout, use_feat_emb=False):
        super().__init__()
        self.fc_in = nn.Sequential()
        self.drop = nn.Dropout(pre_dropout)
        self.dy_ple_layer = DyPLELayer(d_in, n_bins, dy_raw_bin_width, 1)
        self.nlinear = _NLinear(d_in, n_bins, d_feat_emb)
        self.bn0 = nn.BatchNorm1d(d_in) if pre_use_bn else nn.Identity()

    def forward(self, h):
        h = self.dy_ple_layer(h)
        h = self.nlinear(h)
        h = F.relu(h, inplace=True)
        return h

    def init_params(self, x, y):
        self.dy_ple_layer.init_params(x, y)


class ASAGEConv(nn.Module):
    def __init__(
            self,
            n,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=False,
            norm=None,
            activation=None,
    ):
        super().__init__()
        valid_aggre_types = {"mean", "gcn", "max_pool", "mean_pool"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn

        self.fc_pool = _NLinear(n,self._in_src_feats, self._in_src_feats)
        self.fc_neigh = _NLinear(n,self._in_src_feats, out_feats, bias=False)
        self.fc_self = _NLinear(n,self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            if "pool" in self._aggre_type:
                graph.srcdata["h"] = feat_src # F.relu(self.fc_pool(feat_src))
                if self._aggre_type == "max_pool":
                    graph.update_all(msg_fn, fn.max("m", "neigh"))
                if self._aggre_type == "mean_pool":
                    graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class ASAGE(nn.Module):
    def __init__(self, d_in, n_classes=2, **kwargs):
        super().__init__()
        d_hidden = kwargs['d_hidden']

        preprocess = kwargs['preprocess']
        n_bins = kwargs['n_bins']
        use_dyple = kwargs['use_dyple']
        d_feat_emb = kwargs['d_feat_emb']

        pre_n_layers = kwargs['pre_n_layers']
        pre_use_bn = kwargs['pre_use_bn']
        pre_dropout = kwargs['pre_dropout']

        gnn_n_layers = kwargs['gnn_n_layers']
        gnn_use_bn = kwargs['gnn_use_bn']
        gnn_dropout = kwargs['gnn_dropout']
        gnn_agg = kwargs['gnn_agg']
        self.gnn_use_res = kwargs['gnn_use_res']

        self.use_mha = kwargs['use_mha']
        mha_n_layers = kwargs['mha_n_layers']
        mha_n_heads = kwargs['mha_n_heads']
        first_d_in = d_in

        if self.use_mha:
            self.mha = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=mha_n_heads, dropout=0., batch_first=True)

        self.act = getattr(nn, 'ReLU')()
        self.fc_in = nn.Sequential()

        # PLE 相关
        if not use_dyple and preprocess in ['None', 'PLEC']:
            pass
        elif not use_dyple and preprocess in ['PLEM']:
            self.fc_in.append(_NLinear(d_in, n_bins, n_bins))
            self.fc_in.append(self.act)
            self.fc_in.append(nn.Flatten(start_dim=1))
            d_in = d_in * kwargs.get('n_bins')
        elif use_dyple and preprocess in ['None']:
            self.dy_ple_layer = DyPLELayer(d_in, kwargs.get('n_bins'), kwargs.get('dy_raw_bin_width'), 1)
            self.fc_in.append(self.dy_ple_layer)
            self.fc_in.append(_NLinear(d_in, kwargs.get('n_bins'), d_feat_emb))
            self.fc_in.append(self.act)
            self.fc_in_flatten = nn.Sequential(nn.Linear(first_d_in*d_feat_emb, d_hidden),nn.ReLU(),)
            d_in = d_feat_emb
        else:
            raise ValueError(f'use_dyple: {use_dyple} preprocess: {preprocess}')

        # 前置mlp
        if pre_n_layers == 0:
            self.fc_in.append(nn.Identity())
        else:
            for i in range(pre_n_layers):
                if i == 0:
                    self.fc_in.append(nn.Linear(d_in, d_feat_emb))
                else:
                    self.fc_in.append(nn.Linear(d_feat_emb, d_feat_emb))
                self.fc_in.append(nn.ReLU())
                self.fc_in.append(nn.BatchNorm1d(d_feat_emb)) if pre_use_bn else None
                self.fc_in.append(nn.Dropout(pre_dropout))
            d_in = d_feat_emb

        self.gnn_layers = nn.ModuleList()
        self.gnn_fcs = nn.ModuleList()
        for i in range(gnn_n_layers):
            self.gnn_layers.append(ASAGEConv(first_d_in, d_in, d_feat_emb, gnn_agg, activation=self.act,
                                             feat_drop=gnn_dropout,
                                             norm=nn.BatchNorm1d(d_feat_emb) if gnn_use_bn else None))
            self.gnn_fcs.append(nn.Sequential(nn.Linear(first_d_in*d_feat_emb, d_hidden),nn.ReLU(),))
            d_in=d_feat_emb

        self.fc_out = nn.Linear(d_hidden, n_classes)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # 遍历模型的所有模块
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, blocks, h):
        h = self.fc_in(h)
        h_hat = self.fc_in_flatten(h.flatten(1))
        # v = h[:blocks[-1].num_dst_nodes()]
        # 逐层应用GraphSAGE卷积
        h_in = h
        for i, (block, layer,gnn_fc) in enumerate(zip(blocks, self.gnn_layers, self.gnn_fcs)):
            last_h_hat = h_hat
            h = layer(block, h_in)
            h_hat = gnn_fc(h.flatten(1))
            if self.gnn_use_res:
                h_hat = 0.7 * h_hat + 0.3 * last_h_hat[:block.num_dst_nodes()]
            h_in = h

        if self.use_mha:
            v = q = k = h[:blocks[-1].num_dst_nodes()]
            o, _ = self.mha(q, k, v)
            o = self.act(o)
            h = 0.5 * o + 0.5 * h
        # 最后的全连接层
        h = self.fc_out(h_hat)
        return h

if __name__ == '__main__':
    # test GroupSageConv
    # g = dgl.heterograph({
    #     ('u', 'e1', 'u'): ([0, 1], [6, 6]),
    #     ('u', 'e2', 'u'): ([2, 3], [6, 6]),
    #     ('u', 'e3', 'u'): ([4, 5], [6, 6]),
    # })
    # g.ndata['h'] = torch.FloatTensor([[2., 2.], [3., 3.], [7., 7.], [13., 13.], [29., 29.], [47., 47.], [0., 0.]])
    # g.ndata['group'] = torch.FloatTensor([[0], [1], [0], [1], [0], [1], [0]])  # dynamic group
    #
    # g = dgl.to_block(g)
    # gsc = GroupSageConv1(2, 3)
    # h = gsc(g, g.nodes['u'].data['h'])
    # print(g.nodes['u'].data['h'].shape)
    # print(h.shape)

    # gs = GroupSage(2, 2, n_head=2)
    # h = gs(g, g.ndata['h'])
    # print(g.ndata['h'].shape)
    # print(h.shape)

    # g = dgl.to_block(g)
    # gs = GroupSage(2, 2, n_head=2)
    # h = gs([g], g.nodes['_N'].data['feat'])
    # print(g.nodes['u'].data['h'].shape)
    # print(h.shape)

    # 使用真实数据测试
    g = load_graphs('/home/dmj/rhspace/001_GADBench/datasets/yelp')[0][0]
    num_nodes = g.number_of_nodes()
    g.ndata['group'] = F.one_hot(torch.randint(0, 2, (num_nodes,)))

    g = dgl.to_block(g)
    gsc = GroupSageConv2(32, 64)
    h = gsc(g, g.srcdata['feature'])
    print(h.shape)
