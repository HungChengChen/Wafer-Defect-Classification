import argparse
import os
import re

from lightning.lite.utilities.seed import seed_everything


class opts(object):
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--exp_id", type=str, default="default", help="Name of the project."
        )
        self.parser.add_argument(
            "--class_mode",
            type=str,
            default="binary",
            choices=["binary", "multiclass"],
            help="Choose between 'binary' or 'multiclass' classification.",
        )
        self.parser.add_argument(
            "--sampling",
            type=str,
            default="kfold",
            choices=["kfold", "stratified"],
            help="Choose between 'kfold' or 'stratified' sampling.",
        )
        self.parser.add_argument(
            "--n_splits",
            type=int,
            default=5,
            help="For 'kfold' sampling, it represents the number of folds. For 'stratified' sampling, \
            it represents the percentage of the dataset to include in the validation split (1/n_splits).",
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="data/Binary",
            help="Directory containing the data",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="convnext_tiny.fb_in1k",
            help="Name of the model to use (see torchvision.models for options)",
        )
        self.parser.add_argument(
            "--use_pretrained",
            type=bool,
            default=True,
            help="Whether to use pretrained weights.",
        )
        self.parser.add_argument(
            "--devices",
            type=str,
            default="auto",
            help="The devices to use (e.g., '1' for one GPU, '[0,1,2]' for GPUs 0, 1, and 2, 'auto' for all available GPUs).",
        )
        self.parser.add_argument(
            "--accelerator",
            type=str,
            default="cuda",
            help="Set the accelerator type ('cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto')",
        )
        self.parser.add_argument(
            "--epochs", type=int, default=60, help="Number of epochs to train for"
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
        )
        self.parser.add_argument(
            "--eta_min",
            type=float,
            default=1e-5,
            help="Minimum learning rate for the Cosine Annealing scheduler.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random state for reproducibility (use a specific seed or leave as None for a random seed)",
        )
        self.parser.add_argument(
            "--min_batch_size",
            type=int,
            default=32,
            help="Batch size for training and validation",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=256, help="Batch size for testing"
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers for data loading",
        )
        self.parser.add_argument(
            "--gradient_clip_val",
            type=float,
            default=0.5,
            help="Gradient clipping value for the optimizer",
        )

    def parse(self, args=""):
        self.opt = self.parser.parse_args(args if args else [])

        self.opt.log_dir = f"{self.opt.exp_id}/{self.opt.class_mode}_{self.opt.sampling}/{self.opt.model_name}"

        if self.opt.seed is None:
            self.opt.seed = seed_everything()
        else:
            seed_everything(self.opt.seed)

        # 解析設備參數
        if self.opt.devices.isdigit():  # 如果是一個數字，指定使用該數量的GPU
            num_gpus = int(self.opt.devices)
            devices = num_gpus if num_gpus > 0 else "auto"
        elif self.opt.devices in ["auto", "-1"]:  # 使用所有可用的GPU
            devices = "auto"
        else:
            devices = [int(d) for d in re.findall(r"\d+", self.opt.devices)]
        self.opt.devices = devices
        return self.opt

    def set_fold(self, opt, fold_index):
        opt.fold = fold_index
        # opt.log_dir = f"{opt.exp_id}/{opt.class_mode}_{opt.kfold}/{opt.seed}/{opt.model_name}/fold-{opt.fold}"
        opt.log_dir = os.path.join(opt.log_dir, f"fold-{opt.fold}")
