import argparse
import optuna
from pathlib import Path
import collections

from train import main
from parse_config import ConfigParser
from utils.util import write_experiment_finding
from utils.logger import setup_logging
from utils import write_json


class Hyperparameters:
    def __init__(self, trial, exp_name):
        # general
        self.seed = 42
        self.exp_name = exp_name

        # learning progress
        self.lr = trial.suggest_float("lr", low = 1e-5, high = 1e-1)
        self.optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

        # loss coefficient
        # loss = alpha_0 * l_ce + alpha_1 * l_c + alpha_2 * l_gmlc
        alpha0 = trial.suggest_float("alpha0", low = 0.01, high = 1.0)
        alpha1 = trial.suggest_float("alpha1", low = 0.01, high = 1.0)
        alpha2 = trial.suggest_float("alpha2", low = 0.01, high = 1.0)
        self.loss_coefficient = [alpha0, alpha1, alpha2]

        # constrastive loss
        self.lambda_pull = trial.suggest_float("lambda_pull", low = 0.01, high = 10.0)
        self.lambda_push = trial.suggest_float("lambda_push", low = 0.01, high = 10.0)
        self.a0 = trial.suggest_float("a0", low = 5.0, high = 15.0)
        self.a1 = trial.suggest_float("a1", low = 5.0, high = 15.0)
    
    def export(self):
        params = {
            # general
            "seed": self.seed,
            "exp_name": self.exp_name,

            # learning process
            "lr": self.lr,
            "optimizer": self.optimizer,    

            # loss coefficient
            "loss_coefficient": self.loss_coefficient,

            # constrastive loss
            "lambda_pull": self.lambda_pull,
            "lambda_push": self.lambda_push,
            "a0": self.a0,
            "a1": self.a1,
        }
        return params

def update_config(arguments, parameters: dict = {}):
    cfg = ConfigParser.from_args(arguments, options=[])
    cfg["optimizer"]["type"] = parameters["optimizer"]
    cfg["optimizer"]["args"]["lr"] = parameters["lr"]
    cfg["trainer"]["args"]["save_dir"] = parameters["exp_name"]
    cfg["trainer"]["args"]["loss_coefficient"] = parameters["loss_coefficient"]
    cfg["trainer"]["args"]["contrastive_loss"]["lambda_pull"] = parameters["lambda_pull"]
    cfg["trainer"]["args"]["contrastive_loss"]["lambda_push"] = parameters["lambda_push"]
    cfg["trainer"]["args"]["contrastive_loss"]["a0"] = parameters["a0"]
    cfg["trainer"]["args"]["contrastive_loss"]["a1"] = parameters["a1"]
    if parameters["optimizer"] == "SGD":
        # only these params are used in SGD
        cfg["optimizer"]["args"]["weight_decay"] = 2e-4
        cfg["optimizer"]["args"]["momentum"] = 0.9
        cfg["optimizer"]["args"]["nesterov"] = True
    cfg._save_dir=Path(parameters["exp_name"])
    cfg._log_dir=Path(parameters["exp_name"])
    write_experiment_finding(parameters, str(cfg.save_dir), "config")
    setup_logging(cfg._log_dir)
    return cfg

def objective(trial):
    # update trial parameters
    trial_num = trial.number
    exp_name = "out/hypertune/cifar10/ir10/" + str(trial_num)
    Path(exp_name).mkdir(parents=True, exist_ok=True)
    
    parameters = Hyperparameters(trial, exp_name=exp_name)
    cfg = update_config(arguments, parameters.__dict__)

    # start training
    res = main(cfg)
    write_experiment_finding(parameters.export(), exp_name, "parameters")
    return res['best_val_accuracy']

if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description='')
    arguments.add_argument('-c', '--config', type=str, default='configs/hypertune/config_cifar10_ir10.json')
    arguments.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    arguments.add_argument('-d', '--device', default='0', type=str, help='indices of GPUs to enable (default: all)')

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=5)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    
    print("Best hyperparameters:", study.best_params)