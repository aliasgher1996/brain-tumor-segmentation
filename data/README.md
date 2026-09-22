# Data

This project uses the **LGG MRI Segmentation** dataset (Buda et al., 2019). It contains brain MRI slices (FLAIR and related sequences, stored as 3-channel `.tif` files) from lower-grade glioma patients in The Cancer Imaging Archive. Each slice has a manual binary mask of the FLAIR abnormality.

**Download:** https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Unzip the download and place the `kaggle_3m` folder here:

```
data/
└── kaggle_3m/
    ├── TCGA_CS_4941_19960909/
    │   ├── TCGA_CS_4941_19960909_1.tif
    │   ├── TCGA_CS_4941_19960909_1_mask.tif
    │   └── ...
    └── ...
```

To keep the data elsewhere, pass the path to every command, for example: `--opts data.root=/path/to/kaggle_3m`.

## Labels and split

- **Diagnosis label:** a slice is `TUMOR` if its mask has at least one positive pixel, and `NORMAL` otherwise.
- **Split:** `python scripts/prepare_splits.py` writes `data/splits/{train,val,test}.csv`.
  - 70% train, 10.5% validation, 19.5% test.
  - Stratified by diagnosis, with seed 768.
  - Same as in the notebook.

**Note on the default split.** It is made per image slice, as in the notebook, so slices from the same patient can appear in train and test. For a stricter evaluation on unseen patients, set `data.split_by_patient=true`.

## Citation

> Buda M., Saha A., Mazurowski M.A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine* 109:218–225. doi:10.1016/j.compbiomed.2019.05.002
