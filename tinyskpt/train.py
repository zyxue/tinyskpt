from pathlib import Path

import yaml

from tinyskpt import config_types


def _load_config(path: Path) -> config_types.Config:
    return config_types.Config.model_validate(yaml.safe_load(path.read_text()))


def train(config_path: Path | None = None, output_dir: Path | None = None) -> None:
    """Train a transformer.

    Args:
        config_path: path to the config file.
        output_dir: path to the output directory where the model and training
            progress are to be saved.
    """
    output_dir = output_dir or Path.cwd()
    config_path = (
        config_path or (Path(__file__).parent / "data/default_config.yaml").resolve()
    )

    print(f"{output_dir=:}")
    print(f"{config_path=:}")

    config = _load_config(config_path)
    print(f"{config=:}")
    print(f"{config.arch_config.head_size=:}")

    print("TODO: Implement training logic.")
