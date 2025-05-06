import dgl
import numpy as np
import pandas as pd
import torch
import math
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import move_data_to_device
from torch.nn import functional as F

# mycode， 代码根目录写入source，然后就可以同一使用基于该目录的绝对路径
import sys
from pathlib import Path
DIR_SOURCE = str(Path(__file__).resolve().parent.parent)  # 代码根目录写入source，然后就可以同一使用基于该目录的绝对路径
sys.path.append(DIR_SOURCE)
from nn.gnn import MLP, SAGE, SAGEHis, PrePMP, ASAGE
from utils.util import masks_to_indexs, cal_binary_metrics


class LitGNN(LightningModule):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__()
        self.outs = []
        self.model = None #SAGE(d_in, n_classes, **kwargs)
        self.gnn_n_layers = kwargs.get('gnn_n_layers', 1)
        self.lr = lr
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.init_bin_width = kwargs.get('init_bin_width', True)
        self.use_dyple = kwargs['use_dyple']

    # 统一batch的输入输出, subgraph batch or neighbor sampling batch
    def unify_batch(self, batch, mask_key='train_mask'):
        if isinstance(batch, tuple):
            input_nodes, output_nodes, blocks = batch
            x = blocks[0].srcdata['feature']
            y = blocks[-1].dstdata['label']
            mask = slice(None)
        else:
            blocks = [batch] * self.gnn_n_layers
            x = blocks[0].ndata['feature']
            y = blocks[-1].ndata['label']
            mask = blocks[-1].ndata[mask_key]
        return blocks, x, y, mask

    def forward(self, blocks, x):
        return self.model(blocks, x)

    def on_fit_start(self):
        self.trn_idx, self.val_idx, self.tst_idx = masks_to_indexs(self.trainer.datamodule.g)
        if self.init_bin_width and self.use_dyple:
            self.model.dy_ple_embedding.init_params(self.trainer.datamodule.g.ndata['feature'][self.trn_idx],
                                                    self.trainer.datamodule.g.ndata['label'][self.trn_idx])

    def training_step(self, batch, batch_idx):
        blocks, x, y, mask = self.unify_batch(batch, 'train_mask')
        logit = self(blocks, x)
        loss = F.cross_entropy(logit[mask], y[mask])
        self.log('trloss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(y[mask]))
        # if math.isnan(loss.item()):
        #     print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        blocks, x, y, mask = self.unify_batch(batch, 'val_mask')
        logit = self(blocks, x)
        nid = blocks[-1].dstdata[dgl.NID]
        self.outs.append((y, logit, nid))

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            outs = self.outs
            y = torch.cat([x[0] for x in outs])
            logit = torch.cat([x[1] for x in outs])
            nid = torch.cat([x[2] for x in outs])

            loss = F.cross_entropy(logit[self.val_idx], y[self.val_idx])
            self.log('valoss', loss, prog_bar=True, on_step=False, on_epoch=True)

            y = y[torch.argsort(nid)]  # 推理的时候会进行shuffe，在此处进行排序
            logit = logit[torch.argsort(nid)]
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
        logit = self(blocks, x)
        nid = blocks[-1].dstdata[dgl.NID]
        self.outs.append((y, logit, nid))

    def on_test_epoch_end(self):
        outs = self.outs
        y = torch.cat([x[0] for x in outs]).cpu().numpy()
        prob = torch.cat([x[1] for x in outs]).softmax(-1).cpu().numpy()[:, 1]
        nid = torch.cat([x[2] for x in outs])
        y = y[torch.argsort(nid).cpu().numpy()]  # 推理的时候会进行shuffe，在此处进行排序
        prob = prob[torch.argsort(nid).cpu().numpy()]
        print()
        out_dic = cal_binary_metrics(y, prob, self.trn_idx, self.val_idx, self.tst_idx, 'f/', verbose=True)
        self.log_dict(out_dic, prog_bar=False, on_step=False, on_epoch=True,add_dataloader_idx=False)
        self.outs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class LitSAGE(LitGNN):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__(d_in, n_classes, lr, **kwargs)
        self.model = SAGE(d_in, n_classes, **kwargs)


class LitMLP(LitGNN):
    def __init__(self, d_in, n_classes, **kwargs):
        super().__init__(d_in, n_classes, **kwargs)
        self.outs = []
        self.model = MLP(d_in, n_classes, **kwargs)
        self.lr = kwargs.get('lr', 0)
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.init_bin_width = kwargs.get('init_bin_width', True)
        self.use_dyple = kwargs['use_dyple']

    def forward(self, blocks, x):
        return self.model(x)

    def on_fit_start(self):
        self.trn_idx, self.val_idx, self.tst_idx = masks_to_indexs(self.trainer.datamodule.g)
        if self.init_bin_width and self.use_dyple:
            self.model.dy_ple_layer.init_params(self.trainer.datamodule.g.ndata['feature'][self.trn_idx],
                                                self.trainer.datamodule.g.ndata['label'][self.trn_idx])

class LitSAGEHis(LitGNN):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__(d_in, n_classes, lr, **kwargs)
        self.model = SAGEHis(d_in, n_classes, **kwargs)
        self.his_emb = self.model.his_emb

    def on_train_epoch_start(self) -> None:
        self.model.his_emb.copy_(self.his_emb)

    def training_step(self, batch, batch_idx):
        blocks, x, y, mask = self.unify_batch(batch, 'train_mask')
        logit, _ = self(blocks, x)
        loss = F.cross_entropy(logit[mask], y[mask])
        self.log('trloss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y[mask]))
        # if math.isnan(loss.item()):
        #     print(loss)
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
        logit = self(blocks, x)
        nid = blocks[-1].dstdata[dgl.NID]
        self.outs.append((y, logit, nid))


class LitPrePMP(LitGNN):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__(d_in, n_classes, lr, **kwargs)
        self.model = PrePMP(d_in, n_classes, **kwargs)

class LitASAGE(LitGNN):
    def __init__(self, d_in, n_classes, lr, **kwargs):
        super().__init__(d_in, n_classes, lr, **kwargs)
        self.model = ASAGE(d_in, n_classes, **kwargs)