import os
import glob
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_ood.dataset.img as OODDataset


class iNaturalist(Dataset):
    def __init__(self, root: str, image_dirs: str, transform: transforms.Compose):
        """
        iNaturalist dataset loader

        Args:f
            root (str): Data root for inaturalist
            image_dirs (str): txt file containing all the image dirs
            transform (transforms.Compose): Transforms
        """
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(image_dirs, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path, _ = line.strip().split(' ')
                self.img_path.append(os.path.join(root, path))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        ret = {
            "img": sample, 
            "label": 0
        }
        return ret

class OoDDataLoader(DataLoader):
    def __init__(
            self, 
            ood_data,
            data_dir, 
            train_dir = "",
            val_dir = "",
            split = "train", 
            image_size = (32, 32), 
            batch_size = 128
        ):
        # we should mimic the transformation used in the ID dataset
        # the mean and standard deviation must be the same as the ID dataset
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img if img.shape[0] == 3 else img.expand(3, -1, -1)),
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            )
        ])
        if split == "train":
            self.image_dirs = train_dir
            self.shuffle = True
        else:
            self.image_dirs = val_dir
            self.shuffle = False

        if ood_data == "iNaturalist2018":
            self.dataset = iNaturalist(data_dir, self.image_dirs, transform)
        elif ood_data == "places365":
            self.dataset = torchvision.datasets.Places365(data_dir, small=True, split="val", transform=transform)
        elif ood_data == "DTD":
            self.dataset = torchvision.datasets.DTD(data_dir, split="test", transform=transform)
        elif ood_data == "LSUN":
            self.dataset = torchvision.datasets.LSUN(data_dir, classes="test_lmdb.zip", transform=transform)
        elif ood_data == "SVHN":
            self.dataset = torchvision.datasets.SVHN(data_dir, split="test", transform=transform)
        elif ood_data == "TinyImageNet":
            self.dataset = OODDataset.TinyImageNet(
                root = data_dir,
                subset = "test",
                download = "True",
                transform=transform
            )
        elif ood_data == "LSUNCrop":
            self.dataset = OODDataset.LSUNCrop(
                root = data_dir,
                download = "True",
                transform=transform
            )
        elif ood_data == "LSUNResize":
            self.dataset = OODDataset.LSUNResize(
                root = data_dir,
                download = "True",
                transform=transform
            )

        self.val_dataset = None
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def split_validation(self):
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)