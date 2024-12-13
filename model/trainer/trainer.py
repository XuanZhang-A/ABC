import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torchvision.utils import make_grid
from os import makedirs

import utils.data_loader as module_dataloader
from base import BaseTrainer
from model.loss import CenterLossReweighted
from utils import inf_loop, MetricTracker, autocast, use_fp16, GLMC_mixed, generate_embeddings


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
            self, 
            model, 
            criterion, 
            metric_ftns, 
            optimizer, 
            config, 
            data_loader,
            valid_data_loader=None, 
            lr_scheduler=None, 
            len_epoch=None,
            loss_coefficient: list = [1.0, 1.0, 1.0],
            contrastive_loss: dict = {},
            **kwargs
        ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        if self.config.config.get("logging", None) == 'wandb':
            wandb.init(project="experiment_lt", 
                    dir=self.config.config['trainer']['args']['save_dir'],
                    config={
                        "batch_size": self.config.config['data_loader']['args']['batch_size'],
                        "dataset": self.config.config['dataset'],
                        "imbalance_ratio": self.config.config['data_loader']['args']['imb_factor'],
                    }
            )
        
        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)
        print("self.add_extra_info",self.add_extra_info)

        self.train_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.train_loader_bal = config.init_obj('data_loader', 
                                                module_dataloader,
                                                allow_override=True, 
                                                shuffle = False,
                                                balanced = True)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(data_loader)
            self.train_loader_bal = inf_loop(self.train_loader_bal)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
       
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.centerlosses = [CenterLossReweighted(num_classes=self.config["arch"]["args"]["num_classes"], feat_dim=64) for _ in range(3)]
        self.loss_coefficient = loss_coefficient
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        print("train epoch")
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()  # 在base_trainer里面,其实就是model
        self.train_metrics.reset()
        
        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, ((data_lt, target_lt), (data_bal, target_bal)) \
            in enumerate(zip(self.train_loader, self.train_loader_bal)):

            # data preparation
            # long-tailed data
            data_lt_1, data_lt_2 = data_lt
            data_lt_1 = data_lt_1.to(self.device)
            data_lt_2 = data_lt_2.to(self.device)
            target_lt = target_lt.to(self.device)

            # balanced sampled data
            data_bal_1, data_bal_2 = data_bal
            data_bal_1 = data_bal_1.to(self.device)
            data_bal_2 = data_bal_2.to(self.device)
            target_bal = target_bal.to(self.device)

            # find head/tail
            mask_diff = (target_lt != target_bal)
            mixed = GLMC_mixed(
                                data_lt_1,
                                data_lt_2,
                                data_bal_1,
                                data_bal_2,
                                target_lt,
                                target_bal,
                            )
            self.optimizer.zero_grad()

            loss = 0
            with autocast():  # 自动混合精度训练, 降低内存消耗
                outputs_lt = self.model(data_lt_1)
                outputs_bal = self.model(data_bal_1)
                outputs_mixup = self.model(mixed["x_mixup"])
                outputs_cutmix = self.model(mixed["x_cutmix"])

                # ---- LGLA loss ----
                # ID data
                loss_ce, calib_lt = self.criterion(
                    output_features = outputs_lt["feat"][-1], 
                    output_logits = outputs_lt["output"], 
                    target = target_lt,
                    extra_info={"logits":outputs_lt["logits"].transpose(0,1)}
                )
                loss_ce += self.criterion(
                    output_features = outputs_bal["feat"][-1], 
                    output_logits = outputs_bal["output"],
                    target = target_bal,
                    extra_info={"logits":outputs_bal["logits"].transpose(0,1)}
                )[0]
                # mixup data
                loss_mix_lt, calib_mix_lt = self.criterion(
                    output_features = outputs_mixup["feat"][-1], 
                    output_logits = outputs_mixup["output"],
                    target = target_lt,
                    extra_info={"logits":outputs_mixup["logits"].transpose(0,1)}
                )
                loss_mix_bal, calib_mix_bal = self.criterion(
                    output_features = outputs_mixup["feat"][-1], 
                    output_logits = outputs_mixup["output"],
                    target = target_bal,
                    extra_info={"logits":outputs_mixup["logits"].transpose(0,1)}
                )
                loss_ce += mixed["lam"] * loss_mix_lt + (1 - mixed["lam"]) * loss_mix_bal
                # cutmix data
                loss_cut_lt, calib_cut_lt = self.criterion(
                    output_features = outputs_cutmix["feat"][-1], 
                    output_logits = outputs_cutmix["output"],
                    target = target_lt,
                    extra_info={"logits":outputs_cutmix["logits"].transpose(0,1)}
                )
                loss_cut_bal, calib_cut_bal = self.criterion(
                    output_features = outputs_cutmix["feat"][-1], 
                    output_logits = outputs_cutmix["output"],
                    target = target_bal,
                    extra_info={"logits":outputs_cutmix["logits"].transpose(0,1)}
                )
                loss_ce += mixed["lam"] * loss_cut_lt + (1 - mixed["lam"]) * loss_cut_bal

                loss_center = 0
                features = torch.split(outputs_lt["feat"][-1], 64, dim=1)
                features_boundary = torch.split(outputs_mixup["feat"][-1], 64, dim=1)
                for idx, (feature_per_expert, feature_boundary) in enumerate(zip(features, features_boundary)):
                    loss_center += self.centerlosses[idx](
                        feature_i = feature_per_expert, 
                        feature_boundary = feature_boundary, 
                        logits = outputs_mixup["logits"][:, idx, :],
                        labels = target_lt
                    )

                # 3. 相似损失
                z_1, p_1 = outputs_mixup['z'], outputs_mixup['p']
                z_2, p_2 = outputs_cutmix['z'], outputs_cutmix['p']
                loss_sim = self.SimSiamLoss(p_1, z_2) + self.SimSiamLoss(p_2, z_1)

                for coef, _loss in zip(self.loss_coefficient, [loss_ce, loss_center, loss_sim]):
                    loss += coef * _loss

            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', round(loss.item(),4))
            self.train_metrics.update("accuracy", 
                                      self.metric_ftns[0](
                                            outputs_lt["output"], 
                                            target_lt, 
                                            return_length=True)) 
            

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.4f} min group LR: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))

                # write calibs into csv using pandas
                makedirs(f"{self.config._save_dir}/calibs", exist_ok=True)
                for i in range(3):
                    with open(f"{self.config._save_dir}/calibs/calib_lt_{i+1}.csv", "a") as f:
                        for cal in calib_lt[i]:
                            f.write("{:0.4f},".format(cal))
                        f.write("\n")
                    with open(f"{self.config._save_dir}/calibs/calib_mix_lt_{i+1}.csv", "a") as f:
                        for cal in calib_mix_lt[i]:
                            f.write("{:0.4f},".format(cal))
                        f.write("\n")
                    with open(f"{self.config._save_dir}/calibs/calib_cut_lt_{i+1}.csv", "a") as f:
                        for cal in calib_cut_lt[i]:
                            f.write("{:0.4f},".format(cal))
                        f.write("\n")

                self.writer.add_image('input', make_grid(data_lt_1, nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : round(v, 4) for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        if self.config.config.get("logging", None) == 'wandb':
            wandb.log({
                "train_loss": log["loss"],
                "cls_loss": loss_ce, 
                "simsiam_loss": loss_sim,
                "max_lr": max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                "min_lr": min([param_group['lr'] for param_group in self.optimizer.param_groups]),
                "train_accuracy": log["accuracy"],
                "val_loss": log["val_loss"],
                "val_accuracy": log["val_accuracy"],
                "val_many_shot_acc": log["val_many_shot_acc"],
                "val_medium_shot_acc": log["val_medium_shot_acc"],
                "val_few_shot_acc": log["val_few_shot_acc"],
            })
        return log
    
    def _valid_epoch(self, epoch, get_class_acc=False):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # ====modify====
        get_class_acc = True
        if get_class_acc:  # cls_num_list是训练集（即长尾数据集）的
            # 获取每个shot的精度
            train_cls_num_list = np.array(self.train_loader.cls_num_list)
            train_cls_num_list_sort  = np.sort(train_cls_num_list)[::-1]
            many_shot_thresh = train_cls_num_list_sort[self.train_loader.num_classes//3]
            medium_shot_thresh = train_cls_num_list_sort[2*self.train_loader.num_classes//3]
            many_shot = train_cls_num_list > many_shot_thresh
            medium_shot = (train_cls_num_list <= many_shot_thresh) & \
                            (train_cls_num_list >= medium_shot_thresh)
            few_shot = train_cls_num_list < medium_shot_thresh

        confusion_matrix = torch.zeros(self.train_loader.num_classes, 
                                       self.train_loader.num_classes).cuda()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # loss = self.criterion(output, target)
                loss, _ = self.criterion(
                    output_features = output["feat"][-1], 
                    output_logits = output["output"], 
                    target = target,
                    extra_info={"logits":output["logits"].transpose(0,1)}
                )
                if isinstance(output, dict):
                    output = output["output"]

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, 
                                              met(output, target, return_length=True, num_classes=self.train_loader.num_classes))

                for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        val_log = self.valid_metrics.result()
        if get_class_acc:  # 返回不同shot的精度
            acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
            acc = acc_per_class.cpu().numpy() 
            many_shot_acc = acc[many_shot].mean()
            medium_shot_acc = acc[medium_shot].mean()
            few_shot_acc = acc[few_shot].mean() 
            val_log.update({
                "many_shot_acc": round(many_shot_acc,4),
                "medium_shot_acc": round(medium_shot_acc),
                "few_shot_acc": round(few_shot_acc),
            })
        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def SimSiamLoss(self, p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

