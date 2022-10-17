import argparse
import collections
import warnings

import numpy as np
import torch
import os

from pyctcdecode import build_ctcdecoder

from BPE_models.BPE_train import bpe_train, kenlm_path
from hw_asr.utils import prepare_device
from lm_scorer.models.auto import AutoLMScorer as LMScorer

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(7)


def main(config):
    logger = config.get_logger("train")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocab = bpe_train(config)

    text_encoder = config.get_text_encoder(vocab)

    decoder = build_ctcdecoder(
        text_encoder.alphabet,
        kenlm_model_path=kenlm_path(),
        alpha=1.5,
        beta=0.7,
    )
    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder))
    model = model.to(device)  # vocab
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        decoder,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
