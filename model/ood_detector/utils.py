import torch


def knn_score(feas_train, feas, k=10):
    _feas_train = torch.clone(feas_train)
    _feas = torch.clone(feas)

    # Normalize features to unit vectors
    feas_train_norm = _feas_train / _feas_train.norm(dim=1, keepdim=True)
    feas_norm = _feas / _feas.norm(dim=1, keepdim=True)

    # Compute similarity matrix (dot product of normalized vectors)
    sim_matrix = torch.mm(feas_norm, feas_train_norm.t())

    # Get top-k values and indices
    D, I = torch.topk(sim_matrix, k, largest=True, sorted=True)
    scores = D.mean(dim=1)  # Compute mean distance for each query

    return scores