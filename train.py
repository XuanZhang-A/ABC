# -*- coding: UTF-8 -*-   
import argparse
import collections
import torch
import numpy as np

import utils.data_loader as module_dataloader
import model.loss as module_loss
import utils.metrics as module_metric
import model.models as module_arch
import model.trainer as module_trainer
from parse_config import ConfigParser
from utils.lr_scheduler import WarmupMultiStepLR


deterministic = True
if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    print(SEED)

def learning_rate_scheduler(optimizer, config):
    if "type" in config._config["lr_scheduler"]: 
        # CustomLR
        if config["lr_scheduler"]["type"] == "CustomLR": # linear learning rate decay
            lr_scheduler_args = config["lr_scheduler"]["args"]
            gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
            print("Scheduler step1, step2, warmup_epoch, gamma:", 
                  (lr_scheduler_args["step1"], lr_scheduler_args["step2"], 
                   lr_scheduler_args["warmup_epoch"], gamma))
            def lr_lambda(epoch):
                if epoch >= lr_scheduler_args["step2"]:
                    lr = gamma * gamma
                elif epoch >= lr_scheduler_args["step1"]:
                    lr = gamma
                else:
                    lr = 1
                
                """Warmup"""
                warmup_epoch = lr_scheduler_args["warmup_epoch"]
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch
                return lr
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # warmup
        elif config["lr_scheduler"]["type"] == "warmup":
            print("warm up...")
            lr_scheduler_args = config["lr_scheduler"]["args"]
            lr_scheduler = WarmupMultiStepLR(
                optimizer,
                lr_scheduler_args["lr_step"],
                gamma=lr_scheduler_args["lr_factor"],
                warmup_epochs=lr_scheduler_args["warmup_epoch"],
            )

        # cosine
        elif config["lr_scheduler"]["type"] == "cosine":
            print("cosine...")
            lr_scheduler_args = config["lr_scheduler"]["args"]
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=lr_scheduler_args["decay_end"], 
                eta_min=1e-4
            )
        else:
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)  # cosine learning rate decay
    else:
        lr_scheduler = None
    return lr_scheduler


def main(config):
    # logger = config.get_logger('train')

    # ====setup data_loader instances====
    data_loader = config.init_obj('data_loader', module_dataloader)
    valid_data_loader = data_loader.split_validation()

    # ====build model architecture, then print to console====
    model = config.init_obj('arch', module_arch)

    # ====get function handles of loss and metrics====
    loss_class = getattr(module_loss, config["loss"]["type"])
    if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
        criterion = config.init_obj('loss', module_loss, 
                                    cls_num_list=data_loader.cls_num_list, 
                                    num_experts=config["arch"]["args"]["num_experts"])
    else:
        criterion = config.init_obj('loss', module_loss, 
                                    cls_num_list=data_loader.cls_num_list)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # ====build optimizer, learning rate scheduler====
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = learning_rate_scheduler(optimizer, config)

    # 选择一个trainer
    trainer = config.init_obj('trainer', module_trainer,
                    model=model, 
                    criterion=criterion, 
                    metric_ftns=metrics, 
                    optimizer=optimizer,
                    config=config,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler,
                )

    trainer.train()
    res = trainer.get_metrics()
    return res


if __name__ == '__main__':
    # Those won't be record in config
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, 
                      type=str, help='config file path (default: None)')
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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;args;epochs'),
        CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
        CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
        CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--save_period'], type=int, target='trainer;args;save_period'),
        CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
        CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
        CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
        CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
        CustomArgs(['--trainer'], type=str, target='trainer;type') 
    ]
    config = ConfigParser.from_args(args, options)
    print("CUDA Available: ", torch.cuda.is_available())
    main(config)
