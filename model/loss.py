from math import cos
from numpy.core.fromnumeric import sort
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
eps = 1e-7 

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
 
class LGLALoss(nn.Module):
    def __init__(self, cls_num_list=None, num_experts=3, s=30, tau=3.0):
        super().__init__()
        self.base_loss = F.cross_entropy
        # 将类别数量列表转换为张量，并计算类别分布的先验概率
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # 类别数量r
        self.s = s  # 温度参数，用于控制分类结果的“尖锐度”
        self.tau = tau  # 温度参数，用于控制分类结果的“尖锐度”
        self.num_experts = num_experts  # 专家数量
        if self.num_experts <= 2:
            self.tau = 1.0  # 如果只有2个专家，将温度参数设置为1.0
        
        print('loss num_experts: ', self.num_experts)
        print("cls_num_list: ", cls_num_list)
        print('self.tau: ', self.tau)
        self.cls_num_list = torch.tensor(cls_num_list).cuda()
        
        mask_cls = []
        if self.num_experts == 2:
            mask_cls.append(torch.ones_like(self.cls_num_list).bool())
            print(mask_cls[0])
        else:
            # 将各类样本按照类别划分为用于训练局部专家的各个组
            self.region_points = self.get_region_points(self.cls_num_list)
            for i in range(len(self.region_points)):
                mask_cls.append((self.cls_num_list > self.region_points[i][0]) & (self.cls_num_list <= self.region_points[i][1]))
                print('i: ', i)
                print(self.region_points[i][0], self.region_points[i][1])
                print(sum(mask_cls[i]), sum(self.cls_num_list[mask_cls[i]]))
        
        self.mask_cls = mask_cls

    def get_region_points(self, cls_num_list):
        """
        根据样本数量,将数据集分成相等的部分
        """
        region_num = sum(cls_num_list) // (self.num_experts - 1)  # 每个组应该有多少个类
        sort_list, _ = torch.sort(cls_num_list)
        region_points = []
        now_sum = 0
        for i in range(len(sort_list)):
            now_sum += sort_list[i]
            if now_sum > region_num:
                region_points.append(sort_list[i])
                now_sum = 0
        region_points = list(reversed(region_points))
        # assert len(region_points) == self.num_experts - 2

        region_left_right = []
        for i in range(len(region_points)):
            if i == 0:
                region_left_right.append([region_points[i], cls_num_list.max()])
            else:
                region_left_right.append([region_points[i], region_points[i-1]])
        region_left_right.append([0, region_points[len(region_points)-1]])
        # assert len(region_left_right) == self.num_experts - 1

        return region_left_right

    def cal_loss(self, logits, one_hot):
        logits = torch.log(F.softmax(logits, dim=1))
        loss = -torch.sum(logits * one_hot, dim=1).mean()
        return loss

    def cal_one_hot(self, logits, target, mask, ind):

        one_hot = torch.zeros(logits.size()).cuda()
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        with torch.no_grad():
            W_yi = torch.sum(one_hot * logits, dim=1)
            theta_yi = torch.acos(W_yi/self.s).float().cuda() # B

        delta_theta = torch.zeros_like(theta_yi).cuda()
        
        for i in range(self.num_experts-1): 
            theta = theta_yi[mask[i]].mean()
            delta_theta[mask[i]] = theta_yi[mask[i]]-theta

        delta_theta = delta_theta.double()
        delta_theta = torch.where(delta_theta<0.0, 1.0, 1.0 + delta_theta) # B
        delta_theta = delta_theta.float().unsqueeze(1)
        one_hot = one_hot * delta_theta

        return one_hot

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        target_num = self.cls_num_list[target]  # 用于判定目标类别是否处于专家内
        mask = []
        for i in range(len(self.region_points)):
            mask.append((target_num > self.region_points[i][0]) & (target_num <= self.region_points[i][1]))
        
        for ind in range(self.num_experts):  # 对于每个专家（每个组）
            expert_logits = extra_info['logits'][ind]
            one_hot = self.cal_one_hot(expert_logits, target, mask, ind)
            
            if ind != self.num_experts -1:  # 局部的专家
                prior = torch.zeros_like(self.prior).float().cuda() + self.prior.max()
                prior[self.mask_cls[ind]]=self.prior[self.mask_cls[ind]]  # list
                loss += self.cal_loss(expert_logits + torch.log(prior + 1e-9), one_hot)
            else:  # 最后一个，是总体的专家
                loss += self.cal_loss(expert_logits + torch.log(self.prior + 1e-9) * self.tau, one_hot)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, lambda_pull=1.0, lambda_push=1.0, a_0=5.0, a_1=5.0):
        super(ContrastiveLoss, self).__init__()
        self.lambda_pull = lambda_pull
        self.lambda_push = lambda_push
        self.a_0 = a_0
        self.a_1 = a_1

    def forward(self, pull: torch.tensor, push:torch.tensor) -> torch.Tensor:
        loss = self.tanh_loss(pull, push)
        return loss
    
    def tanh_loss(self, pull: torch.tensor, push: torch.tensor) -> torch.Tensor:
        """Proposed Tanh Loss

        Args:
            pull (torch.tensor): Pulling force
            push (torch.tensor): Pushing force

        Returns:
            torch.Tensor: loss
        """
        loss = (torch.exp(self.lambda_pull * pull) - torch.exp(self.lambda_push * push) + torch.exp(torch.tensor(self.a_0, device=pull.device))) / (torch.exp(self.lambda_pull * pull) + torch.exp(self.lambda_push * push) + torch.exp(torch.tensor(self.a_1, device=pull.device)))
        return loss 


class CenterLossReweighted(nn.Module):
    """Center loss.
    code from: 
        https://github.com/KaiyangZhou/pytorch-center-loss
        
    Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLossReweighted, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.contrasitve_loss = ContrastiveLoss()

        self.y = torch.eye(num_classes).cuda()

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def reweight(self, logits, labels):
        """Reweight the center loss with entropy of logits and correctness with labels"""
        entropy = torch.sum(-F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
        correctness = F.cross_entropy(logits, self._one_hot(labels))
        weight = entropy + correctness
        return weight
    
    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert to one hot encoding

        Args:
            labels (torch.Tensor): label of the size B, where B is the batch size
        
        Returns:
            torch.Tensor: one hot encoding of the size [B, N]
        """
        return self.y[labels]

    def distance_calculation(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def forward(self, feature_i, feature_boundary, logits, labels):
        weight = self.reweight(logits, labels)
        pull = (weight * self.distance_calculation(feature_i, labels)).mean()
        push = F.mse_loss(feature_i, feature_boundary)
        
        # prevent nan
        pull = torch.clamp(pull, min=0, max=2.3025)
        push = torch.clamp(push, min=0, max=2.3025)
        
        loss = self.contrasitve_loss(pull, push)

        return loss


class A3LALoss(nn.Module):
    def __init__(self, cls_num_list: list, num_experts: int=3, s: float=30, tau: float=3.0):
        super().__init__()
        self.cls_num_list = torch.tensor(cls_num_list).cuda()
        self.prior =  torch.tensor(np.array(cls_num_list) / np.sum(cls_num_list)).float().cuda()

        self.s = s  # 温度参数，用于控制分类结果的“尖锐度”
        self.num_experts = num_experts  # 专家数量
        self.tau = tau if self.num_experts <= 2 else 1.0  # 温度参数，用于控制分类结果的“尖锐度”
        
        mask_cls = []
        if self.num_experts == 2:
            mask_cls.append(torch.ones_like(self.cls_num_list).bool())
        else:
            # 将各类样本按照类别划分为用于训练局部专家的各个组
            self.region_points = self.get_region_points(self.cls_num_list)
            for i in range(len(self.region_points)):
                mask_cls.append((self.cls_num_list > self.region_points[i][0]) & (self.cls_num_list <= self.region_points[i][1]))
        
        self.mask_cls = mask_cls
    
    def get_region_points(self, cls_num_list):
        """
        根据样本数量,将数据集分成相等的部分
        """
        region_num = sum(cls_num_list) // (self.num_experts - 1)  # 每个组应该有多少个类
        sort_list, _ = torch.sort(cls_num_list)
        region_points = []
        now_sum = 0
        for i in range(len(sort_list)):
            now_sum += sort_list[i]
            if now_sum > region_num:
                region_points.append(sort_list[i])
                now_sum = 0
        region_points = list(reversed(region_points))

        region_left_right = []
        for i in range(len(region_points)):
            if i == 0:
                region_left_right.append([region_points[i], cls_num_list.max()])
            else:
                region_left_right.append([region_points[i], region_points[i-1]])
        region_left_right.append([0, region_points[len(region_points)-1]])

        return region_left_right

    def cal_loss(self, logits, one_hot):
        logits = torch.log(F.softmax(logits, dim=1))
        loss = -torch.sum(logits * one_hot, dim=1).mean()
        return loss

    def cal_one_hot(self, logits, target, mask, ind):

        one_hot = torch.zeros(logits.size()).cuda()
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        with torch.no_grad():
            W_yi = torch.sum(one_hot * logits, dim=1)
            theta_yi = torch.acos(W_yi/self.s).float().cuda() # B

        delta_theta = torch.zeros_like(theta_yi).cuda()
        
        for i in range(self.num_experts-1): 
            theta = theta_yi[mask[i]].mean()
            delta_theta[mask[i]] = theta_yi[mask[i]]-theta

        delta_theta = delta_theta.double()
        delta_theta = torch.where(delta_theta<0.0, 1.0, 1.0 + delta_theta) # B
        delta_theta = delta_theta.float().unsqueeze(1)
        one_hot = one_hot * delta_theta

        return one_hot

    def get_calibration_weight_from_energy_loss(self, x: torch.tensor, targets: torch.tensor, T: float = 1) -> torch.tensor:
        """Get calibration weight from energy loss.

        Args:
            x (torch.tensor): penultimate feature
            target (torch.tensor): target class
            T (float, optional): Temperature. Defaults to 1.

        Returns:
            torch.tensor: Reweighting factor, have the same size as self.cls_num_list
        """
        # calculate energy
        # the lower the energy, the higher the confidence the model has in the prediction
        # energy = (-inf, 0), where -inf is highest confidence and 0 is lowest confidence
        energy = -1 * T * torch.logsumexp(x / T, dim=1)
        energy = energy.detach()
        
        # get energy for each target
        calibrated_weights = torch.zeros_like(self.cls_num_list).float().cuda()
        unique_cls = torch.unique(targets)
        max_energy = -1 *  torch.inf
        for target in unique_cls:
            class_energy = energy[targets == target]
            calibrated_weights[target] = class_energy.mean()
            max_energy = max(max_energy, class_energy.mean())
        
        # if zero is found, replace with maximum value
        # we should never have zero value in the calibration weight
        if torch.any(calibrated_weights == 0):
            # this is to ensure they do not reference to the same memory location
            calibrated_weights[torch.where(calibrated_weights == 0)] = torch.clone(max_energy)

        # softmax will scale the value from 0 to 1 
        # with 0 is the best prediction and 1 is the worst prediction,
        # we then add 1 to the softmax to make sure the value is always greater than 1
        softmax_weights = 1 + F.softmax(calibrated_weights.unsqueeze(0), dim=1).squeeze(0)
        return softmax_weights

    def calibrate_prior(self, feature: torch.tensor, target: torch.tensor):        
        calibrated_weight = self.get_calibration_weight_from_energy_loss(feature, target)
        return calibrated_weight

    def forward(self, output_features, output_logits, target, extra_info=None):
        loss = 0
        target_num = self.cls_num_list[target]  # 用于判定目标类别是否处于专家内
        mask = []
        for i in range(len(self.region_points)):
            mask.append((target_num > self.region_points[i][0]) & (target_num <= self.region_points[i][1]))
        
        features = torch.split(output_features, 64, dim=1)
        weights = []
        for ind, expert_features in zip(range(self.num_experts), features):  # 对于每个专家（每个组）
            expert_logits = extra_info['logits'][ind]
            energy_weight = self.calibrate_prior(expert_features, target)
            weights.append(energy_weight)
            one_hot = self.cal_one_hot(expert_logits, target, mask, ind)
            
            if ind != self.num_experts -1:  # 局部的专家
                prior_calib = torch.pow(self.prior, energy_weight)
                prior = torch.zeros_like(prior_calib).float().cuda() + prior_calib.max()
                prior[self.mask_cls[ind]] = prior_calib[self.mask_cls[ind]]  # list
                loss += self.cal_loss(expert_logits + torch.log(prior_calib + 1e-9), one_hot)
            else:  # 最后一个，是总体的专家
                prior_calib = torch.pow(self.prior, energy_weight)
                loss += self.cal_loss(expert_logits + torch.log(prior_calib + 1e-9) * self.tau, one_hot)
        return loss, weights