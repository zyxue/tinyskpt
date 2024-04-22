"""Attention related nn modules."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    """A single head of self-attention."""

    def __init__(
        self,
        *,
        embed_size: int,
        head_size: int,
        context_length: int,
        dropout_rate: float,
    ) -> None:
        """
        Args:
            embed_size: size of the input embedding vector.
            head_size: size of the attention head.
        """
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms (B, C, E)-shaped input -> (B, C, H)-shaped output.

        B: batch size
        C: context length
        E: embedding size
        H: head size
        """

        _, _, embed_size = x.shape

        # key can be interpreted as what info the embedding is looking for.
        key = self.key(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        # query can be interpreted as what infor the embedding contains.
        query = self.query(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        key_transpose = key.transpose(dim0=1, dim1=2)  # (B, C, H) -> (B, H, C)
        weight = query @ key_transpose  # (B, C, H) @ (B, H, C) -> (B, C, C)
        weight /= embed_size**0.5
        weight = weight.masked_fill(self.tril == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # value can be interpreted as what info the embedding can communicate.
        value = self.value(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        return weight @ value  # (B, C, C) @ (B, C, H) -> (B, C, H)


class MultiHeadAttention(nn.Module):
    """Group of multiple sibngle-headed self-attention."""

    def __init__(
        self,
        embed_size: int,
        context_length: int,
        dropout_rate: float,
        num_heads: int,
    ):
        super().__init__()

        # Note, the head size of single-head attention is derived from
        # embed_size // num_heads, so using multi-head attention does not
        # increase parameter numbers.
        head_size = embed_size // num_heads

        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(
                    embed_size=embed_size,
                    head_size=head_size,
                    context_length=context_length,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_heads)
            ]
        )

        self.project = nn.Linear(head_size * num_heads, head_size * num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out
