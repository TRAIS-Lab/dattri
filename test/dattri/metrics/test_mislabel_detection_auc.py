"""Unit test for mislabel detection and auc calculation."""

import torch
from sklearn import metrics

from dattri.metric import mislabel_detection_auc


class TestMislabelDetection:
    """Test mislabel detection function."""

    def test_mislabel_detection_case_1(self):
        """The first small test for mislabel_detection_auc function."""
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[90:] = 1

        auc, (fpr, tpr, _thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(torch.tensor(auc, dtype=torch.float32), torch.tensor(1.0))
        assert metrics.auc(fpr, tpr) == auc

    def test_mislabel_detection_case_2(self):
        """The second small test for mislabel_detection_auc function."""
        scores = torch.arange(100)
        noise_index = torch.zeros(100)
        noise_index[80:90] = 1

        auc, (fpr, tpr, _thresholds) = mislabel_detection_auc(scores, noise_index)

        assert torch.allclose(
            torch.tensor(auc, dtype=torch.float32),
            torch.tensor(8 / 9),
        )
        assert metrics.auc(fpr, tpr) == auc
