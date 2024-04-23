"""Utilities related to data handling."""

import enum
from typing import Literal

import torch


class Split(enum.Enum):

    TRAIN = "train"
    EVAL = "eval"


class DataHandler:

    def __init__(self, data: torch.Tensor, eval_size: float):
        assert 0 < eval_size < 1

        num_train = int((1 - eval_size) * len(data))
        self.train_data = data[:num_train]
        self.eval_data = data[num_train:]

    def get_batch(
        self,
        split: Split,
        batch_size: int,
        context_length: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.eval_data
        if split == Split.TRAIN:
            data = self.train_data

        # Randomly pick starting indices for a batch.
        indices = torch.randint(
            high=len(data) - context_length,
            size=(batch_size,),
        )

        input_indices, target_indices = [], []
        for i in indices:
            input_indices.append(data[i : i + context_length])
            target_indices.append(data[i + 1 : i + context_length + 1])

        return (
            torch.stack(input_indices).to(device),
            torch.stack(target_indices).to(device),
        )
