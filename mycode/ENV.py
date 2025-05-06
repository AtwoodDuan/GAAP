import sys
from pathlib import Path
import os
DIR_BASE = str(Path(__file__).resolve().parent.parent)

DIR_LOG = f"{DIR_BASE}/logs"
os.makedirs(DIR_LOG, exist_ok=True)
DIR_WANDB_LOG = DIR_LOG
DIR_LIGHTNING_LOG = DIR_LOG + "/lightning_logs"
DIR_HYDRA_LOG = DIR_LOG + "/hydra_logs"

DIR_FRAUD_DATASET = f'{DIR_BASE}/datasets'

from utils.dataloader import MiniFDataModule, MiniGDataModule
from nn.litnn import *

tag_dm_map = {
    'MiniG': MiniGDataModule,
    'MiniF': MiniFDataModule,
    # 'FullG': FullGDataModule,
    # 'SAINT': SAINTDataModule,
    # 'NF': NFDataModule,
    # 'SGFormer': SGFormerDataModule,
}


model_lit_map = {
    'MLP': LitMLP,
    'SAGE': LitSAGE,
    'SAGEHis': LitSAGEHis,
    'PrePMP': LitPrePMP,
    'ASAGE': LitASAGE,
    # 'DyPLEMLP': LitDyPLEMLP,
    # 'NodeFormer': LitNodeFormer1,
    # 'SGFormer': LitSGFormer,
    # 'DGAGNN': LitDGAGNN,
}
