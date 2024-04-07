"""Unit test for mislabel detection and auc calculation"""

import torch
import sklearn.metrics as metrics

from dattri.metrics.metrics import mislabel_detection_auc


class TestMislabelDetection:
    """Test mislabel detection function"""

    def test_mislabel_detection_small_1(self):
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[90:] = 1

        auc, (fpr, tpr, thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(auc, torch.tensor(1.0))
        assert metrics.auc(fpr, tpr) == auc

    def test_mislabel_detection_small_2(self):
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[80: 90] = 1

        auc, (fpr, tpr, thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(auc, torch.tensor(8 / 9))
        assert metrics.auc(fpr, tpr) == auc
    