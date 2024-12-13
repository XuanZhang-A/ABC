import argparse
import torch
import numpy as np
from tqdm import tqdm
import collections

from parse_config import ConfigParser
import model.models as module_arch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_ood.dataset.img as OODDataset
from utils import EmbeddingVisualizer
import utils.data_loader as module_dataloader
import utils.metrics as module_metric

from utils.util import write_experiment_finding
import matplotlib.pyplot as plt


class EvalModel:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config["arch"]["args"]["num_classes"]
        # self.cal_ood = config['ood_data_loader']['args']['ood_data'] != "none"
        self.cal_ood = False

        self.logger = config.get_logger('test')
        self.ood_image_count = 0

        # setup data_loader instances
        self.train_data_loader = config.init_obj(
            'data_loader',
            module_dataloader,
            allow_override=True,
            training=True
        )
        self.val_data_loader = config.init_obj(
            'data_loader',
            module_dataloader,
            allow_override=True,
            training=False,
            shuffle=False
        )
        self.ood_data_loader = config.init_obj(
            'ood_data_loader', 
            module_dataloader,
            allow_override = True, 
            split = "val", 
            image_size = (224, 224) if config["dataset"] == "imagenet" else(32, 32)
        )

        # build model
        self.model = config.init_obj('arch', module_arch) 
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # define Metrics
        self.metric_fns = [getattr(module_metric, met) for met in config['metrics']]
        self.total_metrics = torch.zeros(len(self.metric_fns))
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).to(device=self.device)

        self.get_experiment_info()

    @torch.no_grad()
    def val(self):
        # 1. calculate OOD
        if self.cal_ood:
            self.build_ood_memory()
            self._val_ood()

        # 2. Train set (LT set)
        # self._val(mode="train")

        # 3. Test set (balanced set)
        self._val(mode="val")
        
    
    @torch.no_grad()
    def _val(self, mode=None):
        if mode == "train":
            dataloader = self.train_data_loader
        elif mode == "val":
            dataloader = self.val_data_loader

        self._reset()
        feats_1, feats_2, feats_3= [], [], []
        logits_1, logits_2, logits_3 = [], [], []
        output_logits = []
        targets_all = []
        for data, target in tqdm(dataloader, desc=f"ID:{mode} set"):
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            output = outputs["output"] if isinstance(outputs, dict) else outputs
            feat = outputs["feat"][-2].cpu().numpy() if isinstance(outputs, dict) else None
            logits = outputs["logits"].transpose(0, 1) if isinstance(outputs, dict) else None

            if self.cal_ood and mode=="test":
                # for plotting tsne
                _, C, _, _ = feat.shape
                feats_1.append(feat[:, :C//3, :, :].reshape(feat.shape[0], -1))
                feats_2.append(feat[:, C//3:2*C//3, :, :].reshape(feat.shape[0], -1))
                feats_3.append(feat[:, 2*C//3:, :, :].reshape(feat.shape[0], -1))
                logits_1.append(logits[0].cpu().numpy())
                logits_2.append(logits[1].cpu().numpy())
                logits_3.append(logits[2].cpu().numpy())
                output_logits.append(output.cpu().numpy())
                targets_all.append(target)

                # compute for ood detector
                ood_feat = outputs["feat"][-1]
                ood_score1 = self.ood_detector1(logits = logits[0], features = ood_feat[:, :C//3])
                ood_score2 = self.ood_detector2(logits = logits[1], features = ood_feat[:, C//3:2*C//3])
                ood_score3 = self.ood_detector3(logits = logits[2], features = ood_feat[:, 2*C//3:])
                ood_score = ood_score1 + ood_score2 + ood_score3 # lets do simple add for now
                self.auroc.update(ood_score, torch.zeros_like(ood_score))

            if mode != "train":
                batch_size = data.shape[0]
                for i, metric in enumerate(self.metric_fns):
                    self.total_metrics[i] += metric(output, target) * batch_size
                for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1

        if self.cal_ood:
            feats_1 = feats_1 + self.feats_ood_1
            feats_2 = feats_2 + self.feats_ood_2
            feats_3 = feats_3 + self.feats_ood_3
            logits_1 = logits_1 + self.logits_ood_1
            logits_2 = logits_2 + self.logits_ood_2
            logits_3 = logits_3 + self.logits_ood_3
            output_logits = output_logits + self.output_logits_ood
            targets_all = targets_all + self.targets_ood
        
            # visualize final output
            self.visualize_tsne(
                [output_logits], 
                targets_all, 
                extra_info=mode+"_output_logits")

            # visualize logits
            self.visualize_tsne(
                [logits_1, logits_2, logits_3], 
                targets_all, 
                extra_info=mode+"_logits")

            # visualize features
            self.visualize_tsne(
                [feats_1, feats_2, feats_3], 
                targets_all, 
                extra_info=mode+"_feature")

        if mode != "train":
            self.compute_metrics()

    def get_ood_feat_logits(self):
        """
        Only calculate the ood data's feature and logits (top 1000)
        for plotting tsne
        """
        self.feats_ood_1, self.feats_ood_2, self.feats_ood_3 = [], [], []
        self.logits_ood_1, self.logits_ood_2, self.logits_ood_3 = [], [], []
        self.output_logits_ood = []
        self.targets_ood = []
        
        for data in tqdm(self.ood_data_loader, desc="OOD"):
            if isinstance(data, dict):
                data = data["img"]
            if isinstance(data, list):
                data = data[0]
            data = data.to(self.device)
            target = 999 * torch.ones(data.shape[0]).to(self.device)
            outputs = self.model(data)
            output = outputs["output"]
            feat = outputs["feat"][-2].cpu().numpy()
            logits = outputs["logits"].transpose(0, 1).cpu().numpy()
            self.ood_image_count += data.shape[0]

            _, C, _, _ = feat.shape
            if self.ood_image_count <= 1000:
                # features
                self.feats_ood_1.append(feat[:, :C//3, :, :].reshape(feat.shape[0], -1))
                self.feats_ood_2.append(feat[:, C//3:2*C//3, :, :].reshape(feat.shape[0], -1))
                self.feats_ood_3.append(feat[:, 2*C//3:, :, :].reshape(feat.shape[0], -1))
                self.targets_ood.append(target)
                # logits
                self.logits_ood_1.append(logits[0])
                self.logits_ood_2.append(logits[1])
                self.logits_ood_3.append(logits[2])
                # output_logits
                self.output_logits_ood.append(output.cpu().numpy())
            else: break  # for fast calculation of tsne

    def _reset(self):
        """
        reset the metrics of EvalModel (for the validation of multiple datasets)
        """
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).to(device=self.device)
        self.total_metrics = torch.zeros(len(self.metric_fns))

    def get_experiment_info(self):
        self.train_cls_num_list = np.array(self.train_data_loader.cls_num_list)
        num_of_classes = self.train_cls_num_list.shape[0]

        # get the threshold for many, medium, and few shot
        if num_of_classes <= 100:
            many_shot_thresh = self.train_cls_num_list[self.train_data_loader.num_classes//3]
            medium_shot_thresh = self.train_cls_num_list[2*self.train_data_loader.num_classes//3]
        else:
            many_shot_thresh, medium_shot_thresh = 100, 10

        # get the list of shots
        self.many_shot = self.train_cls_num_list > many_shot_thresh
        self.medium_shot = (self.train_cls_num_list <= many_shot_thresh) & (self.train_cls_num_list >= medium_shot_thresh)
        self.few_shot = self.train_cls_num_list < medium_shot_thresh

        # get the number of parameters
        num_params = sum(p.numel() for p in self.model.parameters())

        # print out the summary of the evaluation
        self.logger.info("Summary of Evaluation")
        self.logger.info(f"In Distribution Dataset: {self.config['data_loader']['args']['data_dir']}")
        self.logger.info(f"Out of Distribution Dataset: {self.config['ood_data_loader']['args']['ood_data']}")
        self.logger.info(f"Class number list: {self.train_cls_num_list}")
        self.logger.info(f"Threshold for many, medium: {many_shot_thresh, medium_shot_thresh}")
        self.logger.info(f"Many Shot Training Image Count: {self.train_cls_num_list[self.many_shot].sum()}")
        self.logger.info(f"Medium Shot Training Image Count: {self.train_cls_num_list[self.medium_shot].sum()}")
        self.logger.info(f"Few Shot Training Image Count: {self.train_cls_num_list[self.few_shot].sum()}")
        if self.cal_ood is True:
            self.logger.info(f"Out of Distribution Dataset: {self.config['test_ood']}")
        self.logger.info(f"Model: {self.config['arch']['type']}")
        self.logger.info(f"Total parameters for Model: {num_params}")


    def visualize_tsne(
        self, 
        features: list[torch.tensor], 
        targets_all: list[torch.tensor], 
        extra_info: str):

        targets_all = torch.cat(targets_all, dim=0).cpu()

        for i, feat in enumerate(features):
            tsne = EmbeddingVisualizer(config=config)
            tsne.fit(feat, targets_all.numpy())
            tsne.visualize(
                400, 
                self.config.whole_name+"_" + extra_info + f"_expert{i}",
                str(self.config._save_dir), 
                image_count=np.array(self.train_cls_num_list)
            )

    def compute_metrics(self):
        acc_per_class = self.confusion_matrix.diag()/self.confusion_matrix.sum(1)
        acc = acc_per_class.cpu().numpy() 
        
        # Save accuracy for each class as a bar chart
        fig, ax = plt.subplots()
        N = acc.shape[0]
        ax.bar(range(N), acc)
        ax.set_xlabel('Class index')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy per Class for {config["dataset"]}')
        plt.savefig(str(config._save_dir / "accuracy_bar_chart.png"))
        plt.close(fig)

        # calculate many_shot, medium_shot, few_shot accuracy
        many_shot_acc = acc[self.many_shot].mean()
        medium_shot_acc = acc[self.medium_shot].mean()
        few_shot_acc = acc[self.few_shot].mean() 

        n_samples = len(self.val_data_loader.sampler)
        log = {
            "name": config["name"]
        }
        log.update({
            met.__name__: self.total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_fns)
        })

        # writing out accuracy
        log.update({
            "many_shot_acc": round(many_shot_acc.item(), 4),
            "medium_shot_acc": round(medium_shot_acc.item(), 4),
            "few_shot_acc": round(few_shot_acc.item(), 4),
        })
        
        for key, value in log.items():
            if isinstance(value, str):
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            else:
                self.logger.info('    {:15s}: {:0.4f}'.format(str(key), value))

        write_experiment_finding(log, str(config._save_dir), "test_metrics_LT")


if __name__ == '__main__':
    # Those won't be record in config
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-id', '--run_id', default=None, type=str,
                    help='run_id, use this to make experimental folder \
                    determinstic for running .sh (default: None)')

    # Those will change the config
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--test_ood'], type=str, target='test_ood'),
        CustomArgs(['--ood_data'], type=str, target='ood_data_loader;args;ood_data'),
        CustomArgs(['--ood_data_dir'], type=str, target='ood_data_loader;args;data_dir')
    ]

    config = ConfigParser.from_args(args, options, train=False)
    evaluation = EvalModel(config)
    evaluation.val()
