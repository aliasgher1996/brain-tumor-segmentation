"""Index the dataset and write data/splits/{train,val,test}.csv.

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --opts data.root=/path/to/kaggle_3m data.split_by_patient=true
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unet_seg.config import REPO_ROOT, base_parser, load_config
from unet_seg.data import LABELS, build_index, split_index

SPLIT_DIR = os.path.join(REPO_ROOT, 'data', 'splits')


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.opts)
    ds = build_index(cfg.data.root)
    train, val, test = split_index(ds, cfg.data.val_test_fraction, cfg.data.test_fraction_of_heldout,
                                   cfg.data.seed, cfg.data.split_by_patient)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    total = len(ds)
    print('{} slices from {} patients'.format(total, ds.patient.nunique()))
    for name, part in (('train', train), ('val', val), ('test', test)):
        part.to_csv(os.path.join(SPLIT_DIR, name + '.csv'), index=False)
        counts = part['diagnosis'].value_counts().to_dict()
        print('{:<5} {:>5} ({:5.2f}%)  {}'.format(name, len(part), 100 * len(part) / total,
                                                  {LABELS[k]: v for k, v in sorted(counts.items())}))
    print('Splits written to', SPLIT_DIR)


if __name__ == '__main__':
    main()
