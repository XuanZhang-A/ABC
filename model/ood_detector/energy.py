import torch
import torch.nn as nn


class Energy(nn.Module):
    def __init__(self, error_rate: float = 0.05):
        super().__init__()
        self.error_rate = torch.tensor(error_rate)
        self.train_set = []
        
    def scoring_function(self, score, T = 1.0):
        energy = -1 * T * torch.logsumexp(score / T, dim=1)
        return energy
    
    def update(self, logits, **kwargs):
        pass

    def fit(self):
        pass
        
    def forward(self, logits, **kwargs):
        E = self.scoring_function(logits)
        return E

    def reset(self):
        self.train_set = []