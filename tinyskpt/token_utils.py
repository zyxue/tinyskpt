"""Utilities related to tokenization."""


class Tokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))

        self._vocab_size = len(chars)

        self._char_to_index = {ch: i for i, ch in enumerate(chars)}
        self._index_to_char = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, sequence: str) -> list[int]:
        return [self._char_to_index[c] for c in sequence]

    def decode(self, indices: list[int]) -> str:
        return "".join([self._index_to_char[i] for i in indices])
