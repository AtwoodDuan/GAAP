import os
import gc
import warnings
from copy import deepcopy
# dgl
# 机器学习
# torch
import torch
# 工程化、自建和其他
import wandb
import hydra
from pprint import pprint
# pl
from lightning import (Trainer)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer, TQDMProgressBar
from lightning.pytorch.loggers.wandb import WandbLogger

# my code
from ENV import DIR_LOG, tag_dm_map, model_lit_map
from utils.util import MyProgressBar, fix_seed, format_time

torch.set_float32_matmul_precision('medium')
os.environ["WANDB_DIR"] = os.path.abspath(DIR_LOG)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

warnings.filterwarnings("ignore")
GADBenchDatasets = ['amazon', 'yelp', 'elliptic', 'reddit', 'weibo',
                    'questions', 'tolokers', 'dgraphfin', 'tfinance', 'tsocial', 'grid_data']


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
        self.model = model_lit_map[model_name](**self.cfg)
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
            gradient_clip_val=1
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
@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def run_one_exp(cfg=None):
    cfg = dict(cfg)
    pprint(cfg)
    # 检测是否在进行代码调试，如果是则将mode设置为offline
    project = 'test0704b'
    mode = 'online'

    if 'PYCHARM_HOSTED' in os.environ:
        mode = 'offline'
        print('Running in debug mode, project name changed to test, mode changed to offline')
    wandb.init(project=project, config=cfg, mode=mode)
    cfg = wandb.config
    exp = Experiment(cfg['model_name'], cfg['data_name'], cfg)
    exp.train()
    exp.test()
    wandb.finish(quiet=True)
    del exp
    gc.collect()  # 清理内存, 清理显存，有的时候发现显存一直增加
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run_one_exp()

