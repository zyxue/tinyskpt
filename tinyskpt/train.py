from pathlib import Path


def train(config_path: Path | None = None, output_dir: Path | None = None) -> None:
    """Train a transformer.

    Args:
        config_path: path to the config file.
        output_dir: path to the output directory where the model and training
            progress are to be saved.
    """
    output_dir = output_dir or Path.cwd()
    config_path = (
        config_path or (Path("__file__").parent / "data/default_config.yaml").resolve()
    )
    print("TODO: Implement training logic.")
    print(f"{output_dir=:}")
    print(f"{config_path=:}")
