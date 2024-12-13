import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import collections
from torchvision import transforms
from torch.utils.data import DataLoader

from parse_config import ConfigParser
import model.models as module_arch
import utils.data_loader as module_dataloader
from utils.util import write_experiment_finding
import pytorch_ood.detector as detector

from pytorch_ood.utils import OODMetrics

class EvalModel:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config["arch"]["args"]["num_classes"]
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
        self.model = self.get_model(returns_feat = True)

        # ood_detectors
        self.build_ood_detectors()
        self.ood_without_fit = ["Logit_ODIN", "Prob_MCD"]

        # metrics
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes).to(device=self.device)
        self.ood_metrics = OODMetrics(device=self.device)

        self.get_experiment_info()
        self.initialize()
    
    def initialize(self):
        self.ood_score = []
        self.ood_target = []
        self.real_target = []
        self.ood_metrics.reset()

    def get_model(self, returns_feat):
        model = self.config.init_obj('arch', module_arch, allow_override=True, returns_feat=returns_feat) 
        checkpoint = torch.load(self.config.resume)
        state_dict = checkpoint['state_dict']
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def val(self):
        # Use every detector to evaluate the OOD
        for detector_name, detector in self.ood_detectors.items():
            # reset the detector
            self.current_detector_name = detector_name
            self.current_detector = detector

            # These are for the detectors that need fitting
            if self.current_detector_name not in self.ood_without_fit:
                if hasattr(self.current_detector, "fit"):
                    print(f"Fitting {self.current_detector_name}")
                    self.current_detector.fit(self.train_data_loader, device="cuda")

            # val of OOD set
            self._val_ood()

            # val of test set (balanced set)
            self._val(mode="val")

            # compute metrics
            self.compute_metrics()

            # reset the evaluation
            self.initialize()

    @torch.no_grad()
    def _val(self, mode=None):
        """
        Evaluate the model on the validation set (ID set)
        """
        self.ood_metrics.reset()  # reset the metrics
        for data, target in tqdm(self.val_data_loader, desc=f"ID:{mode} set"):
            data, target = data.to(self.device), target.to(self.device)
            
            if self.current_detector_name in self.ood_without_fit:  # without fitting
                ood_score = self.current_detector.predict(data)
            else:  # need fitting
                outputs = self.model(data)
                output = outputs["output"]

                # OOD score
                if self.current_detector_name[:4] == "Feat":  # use feature
                    ood_score = self.current_detector.predict_features(outputs["feat"][-1])
                else:  # use logits
                    ood_score = self.current_detector.predict_features(output)

            self.ood_score.append(ood_score)
            self.ood_target.append(torch.ones_like(ood_score))
            self.real_target.append(target)

    def _val_ood(self):
        for data in tqdm(self.ood_data_loader, desc=f"[{self.current_detector_name}] OOD detecting"):
            # preprocess the data
            if isinstance(data, dict):
                data = data["img"]
            if isinstance(data, list):
                data = data[0]
            if isinstance(data, tuple):
                data = data[0]
                data = data.unsqueeze(0)
            data = data.to(self.device)

            if self.current_detector_name in self.ood_without_fit:
                ood_score = self.current_detector.predict(data)
            else:
                outputs = self.model(data)
                output = outputs["output"]
                
                # OOD score
                if self.current_detector_name[:4] == "Feat":
                    ood_score = self.current_detector.predict_features(outputs["feat"][-1])
                else:
                    ood_score = self.current_detector.predict_features(output)
            
            self.ood_score.append(ood_score)
            self.ood_target.append(-1 * torch.ones_like(ood_score))
            self.real_target.append(-1 * torch.ones_like(ood_score))
            

    def build_ood_detectors(self):
        model = self.get_model(returns_feat = False)
        self.ood_detectors = {}
        # ==== Probability Based ====
        # MSP, can run
        # self.ood_detectors["Prob_MSP"] = detector.MaxSoftmax(self.model)
        # MCD, can run
        # self.ood_detectors["Prob_MCD"] = detector.MCD(model)
        # Temperature scaling
        # self.ood_detectors["Prob_T_Scale"] = detector.TemperatureScaling(self.model)
        # # KL matching
        # self.ood_detectors["Prob_KL_Match"] = detector.KLMatching(self.model)
        # # Entropy, can run
        # self.ood_detectors["Prob_Entropy"] = detector.Entropy(self.model)

        # ==== Logits Based ====
        # MaxLogit, can run
        # self.ood_detectors["Logit_MaxLogit"] = detector.MaxLogit(self.model)
        # # OpenMax
        # self.ood_detectors["Logit_OpenMax"] = detector.OpenMax(self.model)
        # # BEO, can run
        # self.ood_detectors["Logit_BEO"] = detector.EnergyBased(self.model)
        # WeightedBEO
        # TODO calculated the weight for W_BEO
        # self.ood_detectors["Logit_WeightedEBO"] = detector.WeightedEBO(self.model,)
        # ODIN, can run
        model = self.get_model(returns_feat = False)
        model.train()
        self.ood_detectors["Logit_ODIN"] = detector.ODIN(model)

        # ==== Feature Based ====
        # NOTE since MoE disable the feature space, we will just skip them
        # # Mahalanobis
        # self.ood_detectors["Feat_Mahalanobis"] = detector.Mahalanobis(self.model)
        # # RMD
        # self.ood_detectors["Feat_RMD"] = detector.RMD(self.model)
        # # ViM
        # self.ood_detectors["Feat_ViM"] = detector.ViM(self.model)
        # # KNN
        # self.ood_detectors["Feat_KNN"] = detector.KNN(self.model)
        # # SHE
        # self.ood_detectors["Feat_SHE"] = detector.SHE(self.model)


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

        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Summary of Evaluation")
        self.logger.info(f"In Distribution Dataset: {self.config['data_loader']['args']['data_dir']}")
        self.logger.info(f"Many Shot Training Image Count: {self.train_cls_num_list[self.many_shot].sum()}")
        self.logger.info(f"Medium Shot Training Image Count: {self.train_cls_num_list[self.medium_shot].sum()}")
        self.logger.info(f"Few Shot Training Image Count: {self.train_cls_num_list[self.few_shot].sum()}")
        self.logger.info(f"Out of Distribution Dataset: {self.config['ood_data_loader']['args']['ood_data']}")
        self.logger.info(f"Model: {self.config['arch']['type']}")
        self.logger.info(f"Total parameters for Model: {num_params}")


    def compute_metrics(self):
        # accuracy for 
        n_samples = len(self.val_data_loader.sampler)
        log = {
            "name": self.config.whole_name, 
            "ood_detector": self.current_detector_name,
            "ood_dataset": self.config['ood_data_loader']['args']['ood_data']
        }
        # concatenate the target & score
        ood_score = torch.cat(self.ood_score, dim=0)
        ood_target = torch.cat(self.ood_target, dim=0).long()
        real_target = torch.cat(self.real_target, dim=0).cpu().int().numpy()

        # Calcualte the overall ood metrics
        self.ood_metrics.update(ood_score, ood_target)
        ood_metrics = self.ood_metrics.compute()
        for name, val in ood_metrics.items():
            log[name] = round(val, 4)

        # Calculate metrics for many, medium, and few shot
        for shot_type, mask in zip(
                ["many_shot", "medium_shot", "few_shot"],
                [self.many_shot, self.medium_shot, self.few_shot]
            ):
            self.ood_metrics.reset()
            shot_mask = mask[real_target] | (ood_target.cpu().numpy() == -1)
            self.ood_metrics.update(ood_score[shot_mask], ood_target[shot_mask])
            shot_metrics = self.ood_metrics.compute()
            for name, val in shot_metrics.items():
                log[f"{shot_type}_{name}"] = round(val, 4)

        # Log the metrics
        for key, value in log.items():
            if isinstance(value, str):
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            else:
                self.logger.info('    {:15s}: {:0.4f}'.format(str(key), value))

        # Save the metrics
        os.makedirs(str(config._save_dir) + f"/{config['ood_data_loader']['args']['ood_data']}", exist_ok=True)
        write_experiment_finding(
            log, 
            str(config._save_dir) + f"/{config['ood_data_loader']['args']['ood_data']}", 
            "test_metrics_" + self.current_detector_name)

        # TODO histogram is not supported in pytorch_ood, fix this later
        # plt = self.ood_metrics.plot_histogram()
        # plt.savefig(str(config._save_dir) + f"/{config['ood_data_loader']['args']['ood_data']}" + "/score_plot.png")


if __name__ == '__main__':
    # These won't be recorded in config
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
    args.add_argument('-o', '--ood', default=False, type=bool,
                      help='Evaluate OoD (default: True)')

    # These will change the config
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--test_ood'], type=str, target='test_ood'),
        CustomArgs(['--ood_data'], type=str, target='ood_data_loader;args;ood_data'),
        CustomArgs(['--ood_data_dir'], type=str, target='ood_data_loader;args;data_dir')
    ]

    config = ConfigParser.from_args(args, options, train=False)
    evaluation = EvalModel(config)
    evaluation.val()
