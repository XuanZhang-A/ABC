from torchvision import datasets, transforms
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .cifar_LT import IMBALANCECIFAR10, IMBALANCECIFAR100
from .autoaug import CIFAR10Policy, Cutout
import torch


class ImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    double_out(可选): 返回两张图
    NOTE: 
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                 num_workers=1, training=True, double_out=False, 
                 imb_type='exp', imb_factor=0.01, balanced=False, 
                 resample_weight=1):
        # ====transform====
        normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),  # LGLA有的aug
            CIFAR10Policy(),  # 强的aug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),  # 强的aug
            normalize,
            ])

        if double_out:
            train_trsfm = TwoCropTransform(train_trsfm)
        
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        
        # ====datasets====
        if training:
            self.dataset = IMBALANCECIFAR100(data_dir, 
                                             train=True, 
                                             download=True, 
                                             transform=train_trsfm, 
                                             imb_type=imb_type, 
                                             imb_factor=imb_factor)
            self.val_dataset = datasets.CIFAR100(data_dir, 
                                                 train=False, 
                                                 download=True, 
                                                 transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR100(data_dir, 
                                             train=False, 
                                             download=True, 
                                             transform=test_trsfm)
            self.val_dataset = None

        # ====统计各个类别的样本数====
        self.num_classes = len(np.unique(self.dataset.targets))
        assert self.num_classes == 100
        cls_num_list = [0] * self.num_classes
        for label in self.dataset.targets:
            cls_num_list[label] += 1
        self.cls_num_list = cls_num_list
        
        # ====是否重新平衡采样====
        balanced_sampler = None
        if balanced:
            cls_weight = 1.0 / (np.array(cls_num_list) ** resample_weight)
            cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
            samples_weight = np.array([cls_weight[t] for t in self.dataset.targets])
            samples_weight = torch.from_numpy(samples_weight).double()
            balanced_sampler = torch.utils.data.WeightedRandomSampler(
                                                    samples_weight, 
                                                    len(samples_weight), 
                                                    replacement=True)
            shuffle = False  # NOTE 采样和shuffle不能共存
        
        # ====初始化参数====
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=balanced_sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


class ImbalanceCIFAR10DataLoader(DataLoader):
    """
    Imbalance Cifar10 Data Loader
    double_out(可选): 返回两张图
    NOTE: 
    """
    def __init__(self, data_dir, batch_size, shuffle=True, 
                 num_workers=1, training=True, double_out=False, 
                 imb_type='exp', imb_factor=0.01, balanced=False, 
                 resample_weight=1):
        # ====transform====
        normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            normalize,
            ])
        if double_out:
            train_trsfm = TwoCropTransform(train_trsfm)
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        
        # ====datasets====
        if training:
            self.dataset = IMBALANCECIFAR10(data_dir, 
                                             train=True, 
                                             download=True, 
                                             transform=train_trsfm, 
                                             imb_type=imb_type, 
                                             imb_factor=imb_factor)
            self.val_dataset = datasets.CIFAR10(data_dir, 
                                                 train=False, 
                                                 download=True, 
                                                 transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR10(data_dir, 
                                             train=False, 
                                             download=True, 
                                             transform=test_trsfm)
            self.val_dataset = None

        # ====统计各个类别的样本数====
        self.num_classes = len(np.unique(self.dataset.targets))
        assert self.num_classes == 10
        cls_num_list = [0] * self.num_classes
        for label in self.dataset.targets:
            cls_num_list[label] += 1
        self.cls_num_list = cls_num_list
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        # ====是否重新平衡采样====
        balanced_sampler = None
        if balanced:
            cls_weight = 1.0 / (np.array(cls_num_list) ** resample_weight)
            cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
            samples_weight = np.array([cls_weight[t] for t in self.dataset.targets])
            samples_weight = torch.from_numpy(samples_weight).double()
            balanced_sampler = torch.utils.data.WeightedRandomSampler(
                                                    samples_weight, 
                                                    len(samples_weight), 
                                                    replacement=True)

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=balanced_sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)




class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
