"""Training, evaluation and prediction loops (same logic as the notebook)."""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from unet_seg.metrics import batch_dice, segmentation_scores


def run_epoch(model, loader, loss_fn, device, optimizer=None, threshold=0.5):
    """One pass over ``loader``; trains when an optimizer is given. Returns (mean loss, mean batch Dice)."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_dice = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, disable=not training):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_dice += batch_dice(y, (torch.sigmoid(logits) > threshold).float())
    n = max(1, len(loader))
    return total_loss / n, total_dice / n


def train(model, train_loader, val_loader, cfg, device, out_dir):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=cfg.train.plateau_patience)
    history, best = [], -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        loss, dice = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'loss': loss, 'dice_score': dice,
                        'eval_loss': val_loss, 'eval_dice_score': val_dice, 'lr': lr})
        print('Epoch {}/{} - loss: {:.4f} - dice_score: {:.4f} - eval_loss: {:.4f} - eval_dice_score: {:.4f} - lr: {:.2e}'
              .format(epoch, cfg.train.epochs, loss, dice, val_loss, val_dice, lr))
        if cfg.train.save_best and val_dice > best:
            best = val_dice
            save(model, os.path.join(out_dir, 'best_model.pt'))
    save(model, os.path.join(out_dir, 'final_model.pt'))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    return history


@torch.no_grad()
def evaluate_test(model, loader, device, mask_threshold=0.5, metric_threshold=0.3):
    """Per-image segmentation metrics (at ``metric_threshold``) and predicted diagnosis
    (any pixel above ``mask_threshold``)."""
    model.eval()
    rows = []
    for x, y in tqdm(loader, leave=False):
        prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
        y = y.numpy()
        for t, p in zip(y, prob):
            s = segmentation_scores(t, (p > metric_threshold).astype(np.float32))
            s['model_diagnosis'] = int((p >= mask_threshold).any())
            rows.append(s)
    return pd.DataFrame(rows)


def save(model, path):
    module = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(module.state_dict(), path)


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
