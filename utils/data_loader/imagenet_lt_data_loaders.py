import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .randaugment import rand_augment_transform


class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label

class ImageNetLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """
    def __init__(self,data_dir, batch_size, shuffle=True, num_workers=1, 
                 training=True, balanced=False, double_out=False, resample_weight=0.2,
                 train_txt=None, val_txt=None, test_txt=None):
        # ====设置transform====
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        randaug_n=2
        randaug_m=10
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ])
        if double_out:
            train_trsfm = TwoCropTransform(train_trsfm)
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # ====建立dataset====
        if training:
            dataset = LT_Dataset(data_dir,  train_txt, train_trsfm)
            val_dataset = LT_Dataset(data_dir, val_txt, test_trsfm)
        else: # test
            # NOTE 这里我改成了val了
            dataset = LT_Dataset(data_dir, val_txt, test_trsfm)
            val_dataset = None
        self.dataset = dataset
        self.val_dataset = val_dataset

        # ====计算cls_num_list====
        self.n_samples = len(self.dataset)
        self.num_classes = len(np.unique(dataset.targets))
        assert self.num_classes == 1000
        cls_num_list = [0] * self.num_classes
        for label in dataset.targets:
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
        
        # ====初始化的参数====
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=balanced_sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        #return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]