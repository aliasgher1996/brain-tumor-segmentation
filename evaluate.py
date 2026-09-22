"""Evaluate a trained U-Net on the test split.

Writes to <output_dir>/evaluation/:
  segmentation_metrics.csv  mean accuracy / precision / recall / Dice / IoU for all, normal and tumour images
  diagnosis_metrics.json    tumour vs normal: accuracy, weighted precision / recall / F1, MCC
  per_image_metrics.csv     one row per test image
  confusion_matrix.png      diagnosis confusion matrix
  predictions.png           MRI | true mask | predicted mask for random tumour images

Usage:
    python evaluate.py                                  # final-epoch model (as in the notebook)
    python evaluate.py --opts evaluation.checkpoint=best
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from unet_seg.config import REPO_ROOT, base_parser, load_config
from unet_seg.data import LABELS, MRISegmentationDataset, build_transforms
from unet_seg.engine import evaluate_test, load, write_json
from unet_seg.metrics import diagnosis_scores
from unet_seg.model import build_model
from unet_seg.plotting import plot_confusion, plot_predictions

SPLIT_DIR = os.path.join(REPO_ROOT, 'data', 'splits')
SEG_COLS = ['accuracy', 'precision', 'recall', 'dice_score', 'IoU']


def main():
    parser = base_parser(__doc__)
    parser.add_argument('--num_examples', type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = cfg.evaluation

    ckpt = os.path.join(cfg.train.output_dir, '{}_model.pt'.format(ev.checkpoint))
    model = load(build_model(cfg), ckpt, device)
    test_df = pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))
    _, eval_tf = build_transforms(cfg)
    test_ds = MRISegmentationDataset(test_df, eval_tf)
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    per_image = pd.concat([test_df, evaluate_test(model, loader, device, ev.mask_threshold, ev.metric_threshold)], axis=1)
    out = os.path.join(cfg.train.output_dir, 'evaluation')
    os.makedirs(out, exist_ok=True)
    per_image.to_csv(os.path.join(out, 'per_image_metrics.csv'), index=False)

    seg = pd.DataFrame({
        'overall': per_image[SEG_COLS].mean(),
        'normal (no tumour)': per_image[per_image.diagnosis == 0][SEG_COLS].mean(),
        'tumour': per_image[per_image.diagnosis == 1][SEG_COLS].mean(),
    }).T
    seg.to_csv(os.path.join(out, 'segmentation_metrics.csv'))
    print('Segmentation metrics (test set):')
    print(seg.round(4).to_string())

    y_true, y_pred = per_image['diagnosis'].to_numpy(), per_image['model_diagnosis'].to_numpy()
    diag = diagnosis_scores(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    diag['confusion_matrix'] = cm.tolist()
    diag['errors'] = int((y_true != y_pred).sum())
    diag['n_test'] = int(len(y_true))
    write_json(diag, os.path.join(out, 'diagnosis_metrics.json'))
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=[LABELS[0], LABELS[1]], zero_division=0))
    print({k: round(float(v), 4) for k, v in diag.items() if k != 'confusion_matrix'})
    plot_confusion(cm, os.path.join(out, 'confusion_matrix.png'))

    # qualitative examples on tumour images
    tumour_idx = np.flatnonzero(y_true == 1).tolist()
    pick = random.Random(cfg.data.seed).sample(tumour_idx, min(args.num_examples, len(tumour_idx)))
    images, trues, preds = [], [], []
    model.eval()
    with torch.no_grad():
        for i in pick:
            x, y = test_ds[i]
            p = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy() > ev.mask_threshold
            images.append(x.permute(1, 2, 0).numpy())
            trues.append(y[0].numpy())
            preds.append(p.astype(np.float32))
    plot_predictions(images, trues, preds, os.path.join(out, 'predictions.png'))
    print('Results written to', out)


if __name__ == '__main__':
    main()
