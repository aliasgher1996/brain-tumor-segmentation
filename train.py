"""Train the U-Net.

Usage:
    python scripts/prepare_splits.py          # once
    python train.py
    python train.py --opts train.epochs=50 train.batch_size=16
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from unet_seg.config import REPO_ROOT, base_parser, load_config
from unet_seg.data import MRISegmentationDataset, build_transforms
from unet_seg.engine import train
from unet_seg.model import build_model
from unet_seg.plotting import plot_history

SPLIT_DIR = os.path.join(REPO_ROOT, 'data', 'splits')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.opts)
    set_seed(cfg.data.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count()

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv'))
    train_tf, eval_tf = build_transforms(cfg)
    batch = cfg.train.batch_size * (n_gpus if n_gpus > 1 else 1)
    train_loader = DataLoader(MRISegmentationDataset(train_df, train_tf), batch_size=batch, shuffle=True,
                              num_workers=cfg.train.num_workers)
    val_loader = DataLoader(MRISegmentationDataset(val_df, eval_tf), batch_size=batch, shuffle=False,
                            num_workers=cfg.train.num_workers)

    model = build_model(cfg).to(device)
    if n_gpus > 1 and cfg.train.data_parallel:
        model = nn.DataParallel(model)
    print('Train on {} samples, validate on {} samples ({} | {} GPU(s))'.format(
        len(train_df), len(val_df), device, n_gpus))

    out_dir = cfg.train.output_dir
    os.makedirs(out_dir, exist_ok=True)
    history = train(model, train_loader, val_loader, cfg, device, out_dir)
    plot_history(pd.DataFrame(history), os.path.join(out_dir, 'training_curves.png'))
    print('Checkpoints, history.csv and training_curves.png written to', out_dir)


if __name__ == '__main__':
    main()
