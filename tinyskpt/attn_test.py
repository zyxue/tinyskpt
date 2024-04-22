import pytest
import torch

from tinyskpt import attn


class TestSingleHeadAttention:
    def test_forward(self) -> None:
        batch_size = 1
        embed_size = 2
        head_size = 3
        context_length = 4
        dropout_rate = 0.1

        module = attn.SingleHeadAttention(
            embed_size=embed_size,
            head_size=head_size,
            context_length=context_length,
            dropout_rate=dropout_rate,
        )

        x = torch.Tensor(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            ]
        )
        actual = module(x)
        assert actual.shape == (batch_size, context_length, head_size)


class TestMultiHeadAttention:
    def test_forward(self) -> None:
        batch_size = 1
        embed_size = 5
        context_length = 4
        num_heads = 2
        dropout_rate = 0.1

        module = attn.MultiHeadAttention(
            embed_size=embed_size,
            context_length=context_length,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
        )

        x = torch.Tensor(
            [
                [
                    [-0.1, -0.2, 0, 0.1, 0.2],
                    [-0.3, -0.4, 0, 0.3, 0.4],
                    [-0.5, -0.6, 0, 0.5, 0.6],
                    [-0.7, -0.8, 0, 0.7, 0.8],
                ],
            ]
        )
        actual = module(x)
        assert actual.shape == (batch_size, context_length, (5 // 2) * 2)
