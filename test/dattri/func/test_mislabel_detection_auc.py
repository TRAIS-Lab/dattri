"""Unit test for mislabel detection and auc calculation."""

import torch

from dattri.metrics.metrics import mislabel_detection_auc
from sklearn import metrics


class TestMislabelDetection:
    """Test mislabel detection function."""

    def test_mislabel_detection_small_1(self):
        """The first small test for mislabel_detection_auc function."""
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[90:] = 1

        auc, (fpr, tpr, thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(auc, torch.tensor(1.0))
        assert metrics.auc(fpr, tpr) == auc

    def test_mislabel_detection_small_2(self):
        """The second small test for mislabel_detection_auc function."""
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[80: 90] = 1

        auc, (fpr, tpr, thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(auc, torch.tensor(8 / 9))
        assert metrics.auc(fpr, tpr) == auc
