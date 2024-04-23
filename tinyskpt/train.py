import json
import os
from pathlib import Path
from typing import Callable, Literal

import fire
import torch
import yaml
from loguru import logger

from tinyskpt import attn, config_types, data_utils, token_utils

_DEFAULT_CONFIG_PATH = (Path(__file__).parent / "config/test_config.yaml").resolve()
_DEFAULT_OUTPUT_DIR = Path.cwd().resolve()
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_config(path: Path) -> config_types.Config:
    return config_types.Config.model_validate(yaml.safe_load(path.read_text()))


def _load_data(data_path: Path) -> str:
    return data_path.read_text()


def _init_model(config: config_types.ArchConfig) -> attn.DecoderTransformer:
    return attn.DecoderTransformer(
        vocab_size=config.vocab_size,
        embed_size=config.embed_size,
        head_size=config.head_size,
        context_length=config.context_length,
        num_heads=config.num_heads,
        ff_hidden_scaler=config.ff_hidden_scaler,
        dropout_rate=config.dropout_rate,
        num_layers=config.num_layers,
    )


@fire.decorators.SetParseFns(
    data_path=Path,
    config_path=Path,
    output_dir=Path,
)
def train(
    data_path: str,
    config_path: str | None = _DEFAULT_CONFIG_PATH,
    output_dir: str | None = _DEFAULT_OUTPUT_DIR,
) -> None:
    """Train a transformer.

    Args:
        data_path: path to the text file to be used for training.
        config_path: path to the config file. Default to the test_config.yaml,
            which is mainly for testing purpose.
        output_dir: path to the output directory where the model and training
            progress are to be saved.
    """
    logger.info(f"{data_path=:}")
    logger.info(f"{config_path=:}")
    logger.info(f"{output_dir=:}")

    text = _load_data(data_path)
    config = _load_config(config_path)

    logger.info(f"Length of dataset in characters: {len(text):,}")
    logger.info(f"{config=:}")

    tokenizer = token_utils.Tokenizer(text)

    config.arch_config.vocab_size = tokenizer.vocab_size

    data_handler = data_utils.DataHandler(
        data=torch.tensor(tokenizer.encode(text), dtype=torch.long),
        eval_size=config.eval_config.eval_size,
    )

    model = _init_model(config.arch_config).to(_DEVICE)
    logger.info(
        f"Number of parameters: {model.num_params:,}",
    )

    batch = data_handler.get_batch(
        data_utils.Split.TRAIN,
        batch_size=2,
        context_length=5,
        device=_DEVICE,
    )

    logger.info(batch)
