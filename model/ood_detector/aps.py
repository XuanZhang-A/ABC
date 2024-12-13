import torch
import torch.nn as nn
import torch.nn.functional as F


class APS(nn.Module):
    def __init__(self, error_rate: float = 0.05, topk: int = 1):
        super().__init__()
        self.error_rate = error_rate
        self.topk = torch.tensor([topk])

    def scoring_function(self, score, label=None):
        score = F.softmax(score, dim=1)
        idx = score.argsort(1, descending=True)
        score_sort = torch.gather(score, 1, idx)
        score_cumsum = score_sort.cumsum(dim=1)
        if label is not None:
            return torch.gather(score_cumsum, 1, idx.argsort())[:, label]
        else:
            return torch.gather(score_cumsum <= self.quartile, 1, idx.argsort())[:, self.topk]
    
    def update(self, logits, label = None, **kwargs):
        # label for in distribution only
        E = self.scoring_function(logits, label)
        self.train_set.append(E)

    def fit(self):
        self.train_set = torch.cat(self.train_set)
        n = self.train_set.shape[0]
        q = torch.ceil((n + 1) * (1 - self.error_rate)) / n
        assert q <= 1.0, "quartile cannot be more than 1, increase number of image"
        qhat = torch.quantile(self.train_set, q.to(dtype=self.train_set.dtype), interpolation="higher")
        self.quartile = qhat

    def forward(self, logits):
        preds = self.scoring_function(logits)
        return preds, preds < self.quartile
