import dataclasses
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


@dataclasses.dataclass(frozen=True)
class EstimatedLoss:
    train: float
    eval: float


@torch.no_grad()
def _estimate_loss_for_one_split(
    *,
    split: data_utils.Split,
    model: attn.DecoderTransformer,
    data_handler: data_utils.DataHandler,
    arch_config: config_types.ArchConfig,
    eval_config: config_types.EvalConfig,
) -> float:
    losses = []
    for k in range(eval_config.num_batches_for_eval):
        inputs, targets = data_handler.get_batch(
            split,
            batch_size=arch_config.batch_size,
            context_length=arch_config.context_length,
            device=_DEVICE,
        )
        _, loss = model(inputs, targets)
        losses.append(loss.item())
    return torch.mean(torch.tensor(losses, dtype=torch.float32)).item()


@torch.no_grad()
def estimate_loss(
    model: attn.DecoderTransformer,
    data_handler: data_utils.DataHandler,
    arch_config: config_types.ArchConfig,
    eval_config: config_types.EvalConfig,
) -> EstimatedLoss:
    model.eval()  # Switch to evaluation mode.
    kwargs = {
        "model": model,
        "data_handler": data_handler,
        "arch_config": arch_config,
        "eval_config": eval_config,
    }
    out = EstimatedLoss(
        train=_estimate_loss_for_one_split(
            split=data_utils.Split.TRAIN,
            **kwargs,
        ),
        eval=_estimate_loss_for_one_split(
            split=data_utils.Split.EVAL,
            **kwargs,
        ),
    )
    model.train()  # Switch back to training model.
    return out


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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_config.learning_rate,
    )

    loss_history = []
    for index in range(config.train_config.max_iters):
        # TODO: add a progress interval to the config.
        logger.info(f"Progress: {index + 1}/{config.train_config.max_iters}")

        if (
            index % config.eval_config.eval_interval == 0
            or index == config.train_config.max_iters - 1
        ):
            loss = estimate_loss(
                model, data_handler, config.arch_config, config.eval_config
            )
            logger.info(
                f"step {index}: train loss {loss.train:.4f}, eval loss {loss.eval:.4f}"
            )
            loss_history.append(
                {
                    "step": index,
                    "train_loss": loss.train,
                    "eval_loss": loss.eval,
                }
            )

        inputs, targets = data_handler.get_batch(
            data_utils.Split.TRAIN,
            batch_size=config.arch_config.batch_size,
            context_length=config.arch_config.context_length,
            device=_DEVICE,
        )

        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
