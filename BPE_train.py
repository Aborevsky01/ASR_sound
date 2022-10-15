import argparse
import collections

from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
import sentencepiece as spm

from hw_asr.utils.parse_config import ConfigParser


def main(config):
    text_encoder = config.get_text_encoder()
    dataloaders = get_dataloaders(config, text_encoder)

    with open(ROOT_PATH / "data" / "all.txt", 'w') as f:
        for batch in dataloaders['train']:
            for line in batch['text']:
                f.write(line + '\n')

    spm.SentencePieceTrainer.train(input=ROOT_PATH / "data" / "all.txt",
                                   model_prefix='m', model_type='bpe', vocab_size=50)


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

    config = ConfigParser.from_args(args)
    main(config)