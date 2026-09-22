"""Quick checks: model output shape and metric definitions. No dataset or GPU needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unet_seg.metrics import segmentation_scores  # noqa: E402


def test_perfect_mask():
    m = np.zeros((8, 8), np.float32)
    m[2:5, 2:5] = 1
    s = segmentation_scores(m, m)
    assert s['accuracy'] == 1.0
    assert s['dice_score'] == pytest.approx(1.0)
    assert s['IoU'] == pytest.approx(1.0)


def test_half_overlap():
    t = np.zeros((4, 4), np.float32)
    p = np.zeros((4, 4), np.float32)
    t[0, :2] = 1
    p[0, 1:3] = 1
    s = segmentation_scores(t, p)
    assert s['dice_score'] == pytest.approx(0.5)
    assert s['IoU'] == pytest.approx(1 / 3)
    assert s['precision'] == pytest.approx(0.5)


def test_empty_masks_match_notebook_convention():
    z = np.zeros((4, 4), np.float32)
    s = segmentation_scores(z, z)
    assert s['dice_score'] == 0.0 and np.isnan(s['precision'])


def test_unet_output_shape():
    torch = pytest.importorskip('torch')
    from unet_seg.model import UNet
    model = UNet(block_sizes=(8, 16, 32, 64)).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 1, 224, 224)


def test_split_proportions():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('sklearn')
    from unet_seg.data import split_index
    ds = pd.DataFrame({'patient': ['p{}'.format(i // 20) for i in range(2000)],
                       'diagnosis': [i % 3 == 0 for i in range(2000)]}).astype({'diagnosis': int})
    tr, va, te = split_index(ds)
    assert len(tr) == 1400 and len(va) + len(te) == 600
    assert abs(len(te) - 390) <= 1
    tr, va, te = split_index(ds, by_patient=True)
    assert not (set(tr.patient) & set(te.patient))
