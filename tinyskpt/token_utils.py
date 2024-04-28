"""Utilities related to tokenization."""

import json
from pathlib import Path


class Tokenizer:
    def __init__(
        self, char_to_index: dict[str, int], index_to_char: dict[int, str]
    ) -> None:
        self._char_to_index = char_to_index
        self._index_to_char = index_to_char

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_index)

    def encode(self, sequence: str) -> list[int]:
        return [self._char_to_index[c] for c in sequence]

    def decode(self, indices: list[int]) -> str:
        return "".join([self._index_to_char[i] for i in indices])

    def save(self, path: Path | str) -> None:
        """Saves the tokenizer to a file."""
        with open(path, "wt") as f:
            json.dump(
                {
                    "char_to_index": self._char_to_index,
                    "index_to_char": self._index_to_char,
                },
                f,
            )

    @staticmethod
    def load(path: Path | str) -> "Tokenizer":
        """Loads the tokenizer from a file."""
        with open(path, "rt") as f:
            data = json.load(f)
            return Tokenizer(
                char_to_index=data["char_to_index"],
                index_to_char={
                    # Note json only allows string as keys, so convert them back
                    # to int.
                    int(k): v
                    for k, v in data["index_to_char"].items()
                },
            )
