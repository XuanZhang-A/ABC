import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
import os.path


class Cifar10Imbanlance(Dataset):
    def __init__(self, imbanlance_rate, num_cls=10, file_path="data/",
                 train=True, transform=None, label_align=True, ):
        self.transform = transform
        self.label_align = label_align
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < imbanlance_rate < 1"
        self.imbanlance_rate = imbanlance_rate

        self.num_cls = num_cls
        self.data = self.produce_imbanlance_data(file_path=file_path, train=train,imbanlance_rate=self.imbanlance_rate)
        self.x = self.data['x']
        self.targets = self.data['y'].tolist()
        self.y = self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.class_list

    def produce_imbanlance_data(self, imbanlance_rate, file_path="/data", train=True):

        train_data = torchvision.datasets.CIFAR10(
            root=file_path,
            train=train,
            download=True,
        )
        x_train = train_data.data
        y_train = train_data.targets
        y_train = np.array(y_train)

        rehearsal_data = None
        rehearsal_label = None

        data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)

        for cls_idx in range(self.num_cls):
            if train:
                num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
                data_percent.append(int(num))
            else:
                num = data_num
                data_percent.append(int(num))
        if train:
            print("imbanlance_ration is {}".format(data_percent[0] / data_percent[-1]))
            print("per class num: {}".format(data_percent))

        self.class_list = data_percent


        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2
            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        return task_split


class Cifar100Imbanlance(Dataset):
    def __init__(self, imbanlance_rate=0.1, file_path="data/cifar-100-python/", num_cls=100, transform=None,
                 train=True):
        self.transform = transform
        assert 0.0 < imbanlance_rate < 1, "imbanlance_rate must 0.0 < p < 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.imbanlance_rate = imbanlance_rate

        if train is True:
            self.data = self.produce_imbanlance_data(self.imbanlance_rate)
        else:
            self.data = self.produce_test_data()
        self.x = self.data['x']
        self.y = self.data['y']
        self.targets = self.data['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path,"test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
        }

        return dataset

    def produce_imbanlance_data(self, imbanlance_rate):

        with open(os.path.join(self.file_path,"train"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_train = dict[b'fine_labels']

        y_train = np.array(y_train)
        data_x = None
        data_y = None

        data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)

        for cls_idx in range(self.num_cls):
            num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
            data_percent.append(int(num))

        self.per_class_num = data_percent
        print("imbanlance ration is {}".format(data_percent[0] / data_percent[-1]))
        print("per class num：{}".format(data_percent))

        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2

            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]

            if data_x is None:
                data_x = tem_data
                data_y = tem_label
            else:
                data_x = np.concatenate([data_x, tem_data], axis=0)
                data_y = np.concatenate([data_y, tem_label], axis=0)

        dataset = {
            "x": data_x,
            "y": data_y.tolist(),
        }

        return dataset