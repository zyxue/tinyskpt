"""Main entrypoint"""

import fire

from tinyskpt import train


def main() -> None:
    fire.Fire(
        {
            "train": train.train,
        }
    )
