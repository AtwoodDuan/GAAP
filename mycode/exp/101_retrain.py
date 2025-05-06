import os
import gc
import warnings
from copy import deepcopy
import sys
from pathlib import Path
# dgl
import dgl.function as fn
import dgl
# 机器学习
# torch
from torch import nn
import torch
import torch as th
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from rtdl_num_embeddings import compute_bins
from torch.nn import Identity, Parameter
from torch.nn.functional import relu, silu, elu, selu, gelu, sigmoid, tanh, softmax, log_softmax, dropout
from dgl import DGLError
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, check_eq_shape
# 工程化、自建和其他
import wandb
import hydra
from pprint import pprint
# pl
from lightning import (Trainer)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer, TQDMProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger
# my code
DIR_SOURCE = str(Path(__file__).resolve().parent.parent)  # 代码根目录写入source，然后就可以同一使用基于该目录的绝对路径，mycode
sys.path.append(DIR_SOURCE)
from ENV import DIR_LOG, tag_dm_map, model_lit_map
from utils.util import MyProgressBar, fix_seed, format_time, cal_binary_metrics, masks_to_indexs
from nn.litnn import LitGNN

torch.set_float32_matmul_precision('medium')
os.environ["WANDB_DIR"] = os.path.abspath(DIR_LOG)
# 只能在py文件里运行, 不能在Notebook运行
current_file_path = __file__
file_name = os.path.basename(current_file_path)
EXP_ID = file_name.split(".")[0] # exp_id = "001"
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings("ignore")
GADBenchDatasets = ['amazon', 'yelp', 'elliptic', 'reddit', 'weibo',
                    'questions', 'tolokers', 'dgraphfin', 'tfinance', 'tsocial', 'grid_data']


class DyPLEC(nn.Module):
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


class NLinear(nn.Module):
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


class DyPLEB(nn.Module):
    def __init__(self, d_in, d_feat_emb, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout):
        super().__init__()
        self.fc_in = nn.Sequential()
        self.drop = nn.Dropout(pre_dropout)
        self.dy_ple_layer = DyPLEC(d_in, n_bins, dy_raw_bin_width, 1)
        self.nlinear = NLinear(d_in, n_bins, d_feat_emb)
        self.bn = nn.BatchNorm1d(d_in) if pre_use_bn else nn.Identity()

    def forward(self, h):
        h = self.dy_ple_layer(h)
        h = self.nlinear(h)
        h = F.relu(h, inplace=True)
        h = self.bn(h)
        return h

    def init_params(self, x, y):
        self.dy_ple_layer.init_params(x, y)


class DyPLEM(nn.Module):
    def __init__(self, d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width, pre_use_bn, pre_dropout, use_feat_emb=False):
        super().__init__()
        self.fc_in = nn.Sequential()
        self.drop = nn.Dropout(pre_dropout)
        self.dy_ple_layer = DyPLEC(d_in, n_bins, dy_raw_bin_width, 1)
        self.nlinear = NLinear(d_in, n_bins, d_feat_emb)
        self.bn0 = nn.BatchNorm1d(d_in) if pre_use_bn else nn.Identity()
        self.linear = nn.Linear(d_in * d_feat_emb, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden) if pre_use_bn else nn.Identity()
        self.use_feat_emb = use_feat_emb

    def forward(self, h):
        h = self.dy_ple_layer(h)
        h = self.nlinear(h)
        feat_h = h = F.relu(h, inplace=True)
        h = self.bn0(h)
        h = self.drop(h)
        h = h.flatten(start_dim=1)
        h = self.linear(h)
        h = F.relu(h, inplace=True)
        h = self.bn1(h)
        if self.use_feat_emb:
            return h, feat_h
        else:
            return h


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
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

    def forward(self, graph, feat, _=None):
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


class SAGE(nn.Module):
    def __init__(self, d_in, n_classes=2, **kwargs):
        super().__init__()
        d_hidden = kwargs['d_hidden']

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
        gnn_agg = kwargs['gnn_agg']
        self.gnn_use_res = kwargs['gnn_use_res']

        self.use_mha = kwargs['use_mha']
        mha_n_layers = kwargs['mha_n_layers']
        mha_n_heads = kwargs['mha_n_heads']
        self.alpha = kwargs['mha_alpha']

        self.act = getattr(nn, 'ReLU')()

        if use_dyple and preprocess == 'None':
            self.embedding0 = DyPLEM(d_in, d_feat_emb, d_hidden, n_bins, dy_raw_bin_width,
                                     pre_use_bn, pre_dropout, False)
        elif not use_dyple and preprocess == 'None':
            self.embedding0 = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                self.act,
                nn.BatchNorm1d(d_hidden) if gnn_use_bn else None
            )


        self.post_agg_layers = nn.ModuleList()
        # self.embeddings = nn.ModuleList()
        for i in range(gnn_n_layers):
            self.post_agg_layers.append(
                SAGEConv(d_hidden, d_hidden, gnn_agg, activation=self.act, feat_drop=gnn_dropout,
                         norm=nn.BatchNorm1d(d_hidden) if gnn_use_bn else None)  # , activation=self.act , residual=True
            )

        self.post_fc_out = nn.Sequential(
            nn.Dropout(gnn_dropout),
            nn.Linear(d_hidden, d_hidden),
            self.act,
            nn.BatchNorm1d(d_hidden) if gnn_use_bn else None
        )

        if self.use_mha:
            self.post_fc_out = nn.Sequential(
                nn.Dropout(gnn_dropout),
                nn.Linear(d_hidden, d_hidden*mha_n_heads),
                self.act,
                nn.BatchNorm1d(d_hidden*mha_n_heads) if gnn_use_bn else None
            )
            self.mha = nn.MultiheadAttention(embed_dim=d_hidden*mha_n_heads, num_heads=mha_n_heads, dropout=0., batch_first=True)
        else:
            mha_n_heads=1
        n_nodes = kwargs['n_nodes']
        self.register_buffer('his_emb', torch.rand((n_nodes, d_hidden*mha_n_heads)))
        self.fc_out = nn.Linear(d_hidden*mha_n_heads, n_classes)

    def forward(self, blocks, feat):
        # 0 layer
        num_dst_nodes = blocks[-1].num_dst_nodes()
        pre_agg_list = []
        h = self.embedding0(feat)
        h0 = h
        pre_agg_list.append(h[:num_dst_nodes])

        post_h = h0
        for i, (block, layer) in enumerate(zip(blocks, self.post_agg_layers)):
            last_h = post_h
            post_h = layer(block, post_h).flatten(1)
            if self.gnn_use_res:
                post_h = 0.7 * post_h + 0.3 * last_h[:block.num_dst_nodes()]
        gnn_out = h = self.post_fc_out(post_h)

        # concat and classification
        # out = self.fc_out(torch.concat([pre_h, post_h],-1))

        if self.use_mha:
            v = k = self.his_emb
            q = h[:blocks[-1].num_dst_nodes()]
            o, _ = self.mha(q, k, v)
            o = self.act(o)
            h = self.alpha * o + (1-self.alpha) * h

        h = self.fc_out(h)
        return h, gnn_out


class LitSAGE(LitGNN):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__(d_in, n_classes, lr, **kwargs)
        self.model = SAGE(d_in, n_classes, **kwargs)
        self.his_emb = self.model.his_emb

    def on_train_epoch_start(self) -> None:
        self.model.his_emb.copy_(self.his_emb)

    def training_step(self, batch, batch_idx):
        blocks, x, y, mask = self.unify_batch(batch, 'train_mask')
        logit, _ = self(blocks, x)
        loss = F.cross_entropy(logit[mask], y[mask])
        self.log('trloss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(y[mask]))
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        blocks, x, y, mask = self.unify_batch(batch, 'val_mask')
        logit, his_emb = self(blocks, x)
        nid = blocks[-1].dstdata[dgl.NID]
        self.outs.append((y, logit, nid, his_emb))

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            outs = self.outs
            y = torch.cat([x[0] for x in outs])
            logit = torch.cat([x[1] for x in outs])
            nid = torch.cat([x[2] for x in outs])
            his_emb = torch.cat([x[3] for x in outs])
            loss = F.cross_entropy(logit[self.val_idx], y[self.val_idx])
            self.log('valoss', loss, prog_bar=True, on_step=False, on_epoch=True)

            y = y[torch.argsort(nid)]  # 推理的时候会进行shuffe，在此处进行排序
            logit = logit[torch.argsort(nid)]
            self.his_emb = his_emb[torch.argsort(nid)]
            # self.model.his_emb.copy_(his_emb)
            y = y.cpu().numpy()
            prob = logit.softmax(-1).cpu().numpy()[:, 1]
            out_dic = cal_binary_metrics(y, prob, self.trn_idx, self.val_idx, self.tst_idx)
            self.log("val_auc", out_dic['val_auc'], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_aps", out_dic['val_aps'], prog_bar=True, on_step=False, on_epoch=True)
            self.log("tst_auc", out_dic['tst_auc'], prog_bar=True, on_step=False, on_epoch=True)
            self.log("tst_aps", out_dic['tst_aps'], prog_bar=True, on_step=False, on_epoch=True)
        self.outs.clear()

    def test_step(self, batch, batch_idx):
        blocks, x, y, _ = self.unify_batch(batch)
        y = blocks[-1].dstdata['label']
        logit, _ = self(blocks, x)
        nid = blocks[-1].dstdata[dgl.NID]
        self.outs.append((y, logit, nid))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_aps',
            }
        }

# Experiment 部分,
# 囊括 pipe_name 和 data_name
# 定义 callback：timer， logger， trainer
# 定义 train过程和test过程
class Experiment:
    def __init__(self, model_name, data_name, cfg):
        self.model_name = model_name
        self.data_name = data_name
        self.cfg = cfg
        self.cfg['device'] = cfg.get('device', 'cuda')
        self.device = cfg.get('device', 'cuda')
        if self.device == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get('device_id', 0))
        fix_seed(cfg['seed'])

        # 在 datamodule 中读取数据，特征预处理, 定义dataloader
        self.dm = tag_dm_map[cfg['loader_type']](**self.cfg)
        # 定义模型，依赖于数据形式
        self.cfg['d_in'] = self.dm.d_in
        self.cfg['n_classes'] = self.dm.n_classes
        self.cfg['n_nodes'] = self.dm.g.number_of_nodes()
        print()
        print(f"Model: {model_name}, Data: {data_name}, Loader: {cfg['loader_type']}, Preprocess: {cfg['preprocess']}")
        print(f"d_in: {self.cfg['d_in']}, n_classes: {self.cfg['n_classes']}")
        print(f"batch_size: {self.cfg['bs']}, max_epochs: {self.cfg['max_epochs']}, patience: {self.cfg['patience']}")
        print()
        self.model = LitSAGE(**self.cfg)
        # 定义wandb_run
        self.cfg['dataset'] = data_name
        wandb.config.update({
            'd_in': cfg['d_in'],
            'n_classes': cfg['n_classes'],
            'dataset': cfg['dataset']
        })

    def train(self):
        # 进行训练前的设置，timer，checkpoint，early_stopping，logger
        timer = Timer()
        monitor_metric = 'val_aps' if self.data_name in GADBenchDatasets else 'val_aps'
        checkpoint_callback = ModelCheckpoint(monitor=monitor_metric, mode='max', verbose=False, save_top_k=1)
        early_stopping = EarlyStopping(monitor=monitor_metric, mode='max', verbose=False, patience=self.cfg['patience'])
        logger = WandbLogger(save_dir=DIR_LOG)
        progress_bar = MyProgressBar() if 'PYCHARM_HOSTED' in os.environ else TQDMProgressBar()

        self.trainer = Trainer(
            accelerator=self.device,
            max_epochs=self.cfg['max_epochs'],
            logger=logger,
            callbacks=[progress_bar, checkpoint_callback, early_stopping, timer],
            gradient_clip_val=10
        )
        self.trainer.fit(self.model, self.dm)
        print(f'training time elapsed {format_time(timer.time_elapsed("train"))}')

    def test(self):
        # 读取最优checkpoint 并且进行推理
        pth = self.trainer.checkpoint_callback.best_model_path
        print("Evaluating model in", pth)
        self.model.load_state_dict(torch.load(pth)['state_dict'])
        self.model = self.model.to(self.device)
        self.trainer.test(self.model, self.dm.val_dataloader(), verbose=False)


# run_one_exp
@hydra.main(config_path="../../config/SAGE_MiniF_DyPLE_MHA", config_name="yelp.yaml", version_base=None)
def run_one_exp(cfg=None):
    # 检测是否在进行代码调试，如果是则将mode设置为offline
    torch.set_num_threads(20)
    project = 'GAAP'
    mode = 'offline' if cfg['nowandb'] == True else 'online'# 如果您熟悉wandb，可以将mode设置为online，这样在wandb上可以实时看到实验的进度

    if 'PYCHARM_HOSTED' in os.environ:
        mode = 'offline'
        print('Running in debug mode, project name changed to test, mode changed to offline')
    cfg = dict(cfg)
    wandb.init(project=project, config=cfg, mode=mode, name=EXP_ID)

    cfg = wandb.config
    for key, value in cfg.items():
        print(f"{key}: {value}")
    exp = Experiment(cfg['model_name'], cfg['data_name'], cfg)
    exp.train()
    exp.test()
    wandb.finish(quiet=True)
    del exp
    gc.collect()  # 清理内存, 清理显存，有的时候发现显存一直增加
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run_one_exp()



