import argparse
import collections
import warnings

import numpy as np
import sentencepiece as spm
import torch
import os

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from string import ascii_lowercase
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # text_encoder
    data_path = ROOT_PATH / "test_data" / "transcriptions"
    with open(ROOT_PATH / "test_data" / "all.txt", 'w') as f:
        for file in os.listdir(data_path):
            with open(data_path / file) as infile:
                text = infile.read().lower()
                f.write(text)

    spm.SentencePieceTrainer.train(input=ROOT_PATH / "test_data" / "all.txt",
                                   model_prefix='m', model_type='bpe', vocab_size=50)
    BPE = spm.SentencePieceProcessor()
    BPE.load('m.model')
    vocab = ['_'] + [BPE.id_to_piece(id) for id in range(BPE.get_piece_size())]
    vocab = list(set(vocab).union(set(list(ascii_lowercase))))
    vocab.append('')
    text_encoder = config.get_text_encoder(vocab)

    # setup data_loader instances
    audio_path = ROOT_PATH / "test_data" / "audio"
    transc_path = ROOT_PATH / "test_data" / "transcriptions"
    with (transc_path / "84-121550-0000.txt").open() as f:
        transcription = f.read().strip()
    data = [
        {
            "path": str(audio_path / "84-121550-0000.flac"),
            "text": transcription
        }
    ]
    config['data']['train']['datasets'][0]['args']['data'] = data
    config['data']['val']['datasets'][0]['args']['data'] = data
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(vocab))
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
        BPE,
        vocab,
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
