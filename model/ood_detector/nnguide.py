import torch
import torch.nn as nn

from model.ood_detector.utils import knn_score


class NNGuide(nn.Module):
    def __init__(self, knn = 10):
        """Implementation from NNguide 
        https://github.com/roomo7time/nnguide/blob/main/ood_detectors/nnguide.py

        Args:
            knn (int, optional): KNN value. Defaults to 10.
        """
        super(NNGuide, self).__init__()
        self.knn = knn
        self.gt_class_frequency = {}
        self.logits = []
        self.features = []
    
    def update(self, logits, features, **kwargs):        
        self.logits.append(logits)
        self.features.append(features)
    
    def fit(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.features = torch.cat(self.features, dim=0)
        confs = torch.logsumexp(self.logits, dim=1)
        self.scaled_feas = self.features.reshape(self.features.shape[0], -1) * confs[:, None]
        self.reset()
    
    def forward(self, logits, features, **kwargs):
        # get top knn similarity matrix
        confs = torch.logsumexp(logits, dim=1)
        guidances = knn_score(self.scaled_feas, features.reshape(features.shape[0], -1), k=self.knn)
        scores = guidances * confs
        return scores

    def reset(self):
        # just to clear everything
        # since we are going to do this everytime in val
        self.logits = []
        self.features = []
        self.gt_class_frequency = {}