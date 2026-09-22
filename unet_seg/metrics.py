"""Segmentation and diagnosis metrics, computed as in the notebook.

Segmentation metrics are computed per image on binarised masks and then averaged:
  pixel accuracy, precision, recall, Dice = 2|A∩B| / (|A|+|B|), IoU = |A∩B| / |A∪B|.
Note (kept from the notebook): for an image whose true and predicted masks are both empty,
Dice and IoU are 0 and precision/recall are undefined (NaN, skipped when averaging).
Tumour-positive ("diagnosed") images are therefore reported separately.

Diagnosis: an image is predicted TUMOR when its predicted mask has any positive pixel.
"""
import numpy as np

EPS = 1e-8


def segmentation_scores(y_true, y_pred):
    """y_true, y_pred: binary numpy arrays of one mask."""
    inter = float((y_true * y_pred).sum())
    t, p = float(y_true.sum()), float(y_pred.sum())
    return {
        'accuracy': float((y_true == y_pred).mean()),
        'precision': inter / p if p > 0 else np.nan,
        'recall': inter / t if t > 0 else np.nan,
        'dice_score': 2 * inter / (t + p + EPS),
        'IoU': inter / (t + p - inter + EPS),
    }


def batch_dice(y_true, y_pred):
    """Dice over a whole batch, used for the per-epoch training/validation log."""
    return float((2 * (y_true * y_pred).sum() + EPS) / ((y_true + y_pred).sum() + EPS))


def diagnosis_scores(y_true, y_pred):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy_score': accuracy_score(y_true, y_pred), 'precision_score': p, 'recall_score': r,
            'f1_score': f, 'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)}
