import dataclasses
import json
from pathlib import Path
from typing import TypedDict

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


def _make_tokenizer(text: str) -> token_utils.Tokenizer:
    chars = sorted(list(set(text)))
    return token_utils.Tokenizer(
        char_to_index={ch: i for i, ch in enumerate(chars)},
        index_to_char={i: ch for i, ch in enumerate(chars)},
    )


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
    train_loss: float
    eval_loss: float


@dataclasses.dataclass(frozen=True)
class StepLoss(TypedDict):
    step: int
    train_loss: float
    eval_loss: float


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
def _estimate_loss(
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
        train_loss=_estimate_loss_for_one_split(
            split=data_utils.Split.TRAIN,
            **kwargs,
        ),
        eval_loss=_estimate_loss_for_one_split(
            split=data_utils.Split.EVAL,
            **kwargs,
        ),
    )
    model.train()  # Switch back to training model.
    return out


def _train_loop(
    model: attn.DecoderTransformer,
    optimizer: torch.optim.AdamW,
    data_handler: data_utils.DataHandler,
    config: config_types.Config,
) -> list[StepLoss]:
    """Runs the training loop and returns loss history."""
    loss_history = []
    for index in range(config.train_config.max_iters):
        # TODO: add a progress interval to the config.
        logger.info(f"Progress: {index + 1}/{config.train_config.max_iters}")

        if (
            index % config.eval_config.eval_interval == 0
            or index == config.train_config.max_iters - 1
        ):
            loss = _estimate_loss(
                model, data_handler, config.arch_config, config.eval_config
            )
            logger.info(
                f"step {index}: "
                f"train loss {loss.train_loss:.4f}, "
                f"eval loss {loss.eval_loss:.4f}"
            )
            loss_history.append(
                StepLoss(
                    step=index,
                    train_loss=loss.train_loss,
                    eval_loss=loss.eval_loss,
                )
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

    return loss_history


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

    tokenizer = _make_tokenizer(text)

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

    loss_history = _train_loop(model, optimizer, data_handler, config)

    logger.info("Saving tokenizer ...")
    tokenizer.save(output_dir / "tokenizer.json")
    logger.info("Saving model ...")
    torch.save(model, output_dir / "model.pt")
    logger.info("Saving training results...")
    (output_dir / "loss_history.json").write_text(json.dumps(loss_history))
