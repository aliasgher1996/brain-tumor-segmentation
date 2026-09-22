"""Dataset indexing, train/val/test split, augmentation and the PyTorch Dataset.

Expected layout (LGG MRI Segmentation dataset, Kaggle "kaggle_3m" folder):

    kaggle_3m/
      TCGA_CS_4941_19960909/
        TCGA_CS_4941_19960909_1.tif
        TCGA_CS_4941_19960909_1_mask.tif
        ...
"""
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

LABELS = {0: 'NORMAL', 1: 'TUMOR'}


def load_array(path):
    """Read a .tif image or mask as float32 scaled to [0, 1]."""
    return np.array(Image.open(path)).astype(np.float32) / 255.0


def build_index(root):
    """One row per MRI slice: id, patient, image_path, mask_path, diagnosis (1 = tumour in mask)."""
    paths = glob.glob(os.path.join(root, '*', '*.tif'))
    if not paths:
        raise FileNotFoundError('No .tif files found under {}/<patient>/'.format(root))
    rows = []
    for p in paths:
        patient = os.path.basename(os.path.dirname(p))
        parts = os.path.basename(p).split('_')
        is_mask = parts[-1] == 'mask.tif'
        slice_id = int(parts[-2] if is_mask else parts[-1].replace('.tif', ''))
        rows.append({'id': slice_id, 'patient': patient, 'path': p, 'is_mask': int(is_mask)})
    df = pd.DataFrame(rows)
    images = df[df.is_mask == 0].rename(columns={'path': 'image_path'}).drop(columns='is_mask')
    masks = df[df.is_mask == 1].rename(columns={'path': 'mask_path'}).drop(columns='is_mask')
    ds = images.merge(masks, on=['id', 'patient'], how='left')
    missing = ds['mask_path'].isna().sum()
    if missing:
        raise ValueError('{} images have no matching *_mask.tif'.format(missing))
    ds['diagnosis'] = [int(load_array(m).max() > 0) for m in ds['mask_path']]
    return ds.sort_values(['patient', 'id']).reset_index(drop=True)


def split_index(ds, heldout=0.30, test_of_heldout=0.65, seed=768, by_patient=False):
    """70 / 10.5 / 19.5 % split, stratified by diagnosis (as in the notebook).

    With ``by_patient=True`` whole patients are assigned to a subset instead, stratified by
    whether the patient has any tumour slice, so slices of one patient never leak across subsets.
    """
    from sklearn.model_selection import train_test_split
    if not by_patient:
        train, rest = train_test_split(ds, test_size=heldout, stratify=ds['diagnosis'], random_state=seed)
        val, test = train_test_split(rest, test_size=test_of_heldout, stratify=rest['diagnosis'], random_state=seed)
    else:
        pat = ds.groupby('patient')['diagnosis'].max()
        tr_p, rest_p = train_test_split(pat.index, test_size=heldout, stratify=pat.values, random_state=seed)
        rest_lab = pat.loc[rest_p]
        va_p, te_p = train_test_split(rest_p, test_size=test_of_heldout, stratify=rest_lab.values, random_state=seed)
        train, val, test = (ds[ds.patient.isin(p)] for p in (tr_p, va_p, te_p))
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


def build_transforms(cfg):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    size, aug = cfg.data.image_size, cfg.augmentation
    train_tf = A.Compose([
        A.Resize(size, size, p=1.0),
        A.RandomBrightnessContrast(p=aug.brightness_contrast_p),
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        A.VerticalFlip(p=aug.vertical_flip_p),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([A.Resize(size, size, p=1.0), ToTensorV2()])
    return train_tf, eval_tf


class MRISegmentationDataset:
    """Returns (image [3,H,W] float in [0,1], mask [1,H,W] float in {0,1})."""

    def __init__(self, df, transform):
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image, mask = load_array(self.image_paths[i]), load_array(self.mask_paths[i])
        out = self.transform(image=image, mask=mask)
        return out['image'], out['mask'].unsqueeze(0)
