import torch
import torch.nn.functional as F
import numpy as np
from skimage.filters import gaussian as gblur

def create_ood_noise(noise_type, ood_num_examples, num_to_avg):
    if noise_type == "Gaussian":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.float32(np.clip(
            np.random.normal(size=(ood_num_examples * num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Rademacher":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(ood_num_examples * num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Blob":
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * num_to_avg, 32, 32, 3)))
        for i in range(ood_num_examples * num_to_avg):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=None)
            ood_data[i][ood_data[i] < 0.75] = 0.0

        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    return ood_data


def generate_embeddings(features: list[torch.Tensor], 
                        layer_selection: list[torch.Tensor]):
    """
    选取特定层数的embedding并进行处理
    """
    max_height, max_width = 0, 0
    features = [features[i] for i in layer_selection]
    for feature in features:
        _, _, height, width = feature.shape
        # Update max_height and max_width if the current feature's dimensions are larger
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
    embeddings_for_each_experts = []
    for i in range(3):
        embedding = []
        for feature in features:
            _, C, _, _ = feature.shape
            startC = i * int(C/3)
            endC = (i+1) * int(C/3)
            feat = F.interpolate(feature[:, startC:endC, :, :], size=(max_height, max_width), mode='bilinear', align_corners=True)
            feat = feature_pooler(feat)
            embedding.append(feat)
        embedding = torch.cat(embedding, dim=1)
        embeddings_for_each_experts.append(embedding)
    return embeddings_for_each_experts