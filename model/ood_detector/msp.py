import torch
import torch.nn as nn
import torch.nn.functional as F


class MSP(nn.Module):
    def __init__(self, error_rate: float = 0.05):
        super().__init__()
        self.error_rate = torch.tensor(error_rate)
        self.train_set = []
        
    def scoring_function(self, score):
        return F.softmax(score, dim=1).max(dim=1)[0]
    
    def update(self, logits, **kwargs):
        E = self.scoring_function(logits)
        self.train_set.append(E)

    def fit(self):
        self.train_set = torch.cat(self.train_set)
        n = self.train_set.shape[0]
        assert self.error_rate >= 1 / (n+1), "Number of calibration sample need to be increased"
        q = torch.ceil((n + 1) * (1 - self.error_rate.to(device=self.train_set.device))) / n
        qhat = torch.quantile(self.train_set, q.to(dtype=self.train_set.dtype, device=self.train_set.device))
        self.quartile = qhat
        
    def forward(self, logits, **kwargs):
        E = self.scoring_function(logits)
        return E

    def reset(self):
        self.train_set = []