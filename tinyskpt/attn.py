"""Attention related nn modules.

Symbols:
    B: batch size.
    C: context length.
    E: embedding size.
    H: head size.
    V: vocabulary size.
"""

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
        """Transforms (B, C, E)-shaped input -> (B, C, H)-shaped output."""

        # Note, the context_length (C) can be smaller than the one set during
        # training at the beginning of inference, i.e. when predicting the first
        # C words.
        _, context_length, embed_size = x.shape

        # Key can be interpreted as what info a token contains.
        key = self.key(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        # Query can be interpreted as what info a token is looking for.
        query = self.query(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        key_transpose = key.transpose(dim0=1, dim1=2)  # (B, C, H) -> (B, H, C)
        weight = query @ key_transpose  # (B, C, H) @ (B, H, C) -> (B, C, C)
        weight /= embed_size**0.5
        weight = weight.masked_fill(
            self.tril[:context_length, :context_length] == 0, float("-inf")
        )
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # Value can be interpreted as what info a token can communicate.
        value = self.value(x)  # (B, C, E) @ (E, H) -> (B, C, H)

        out = weight @ value  # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out


class MultiHeadAttention(nn.Module):
    """Group of multiple sibngle-headed self-attention."""

    def __init__(
        self,
        *,
        embed_size: int,
        head_size: int,
        context_length: int,
        dropout_rate: float,
        num_heads: int,
    ):
        super().__init__()

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

        # The linear layer on top of concatenated heads (described in Figure 2
        # in the attention paper).
        self.linear = nn.Linear(head_size * num_heads, head_size * num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out


class FeedForward(nn.Module):
    """Just a simple feed-forward sublayer."""

    def __init__(
        self,
        *,
        input_size: int,
        scaler: int,
        output_size,
        dropout_rate: float,
    ):
        """
        Args:
            input_size: size of the input vector.
            dropout_rate: dropout rate.
            scaler: scalor for calculating the size of the hidden layer: scaler
                * input_size."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, scaler * input_size),
            nn.ReLU(),
            nn.Linear(scaler * input_size, output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class AttentionLayer(nn.Module):
    """Attention layer is composed of a sublayers.

    - Multi-head attention sublayer
    - Feed-forward sublayer
    """

    def __init__(
        self,
        *,
        embed_size: int,
        head_size: int,
        context_length: int,
        dropout_rate: float,
        num_heads: int,
        ff_hidden_scaler: int,
    ) -> None:
        """
        Args:
            embed_size: size of the input embedding vector.
            context_length: length of the context.
            dropout_rate: dropout rate.
            num_heads: number of heads in multi-head attention.
            ff_hidden_scaler: scaler for calculating the the hidden layer size in the
                feedforward sublayer.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            embed_size=embed_size,
            head_size=head_size,
            context_length=context_length,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
        )

        input_size = head_size * num_heads

        self.feed_forward = FeedForward(
            input_size=input_size,
            scaler=ff_hidden_scaler,
            output_size=embed_size,
            dropout_rate=dropout_rate,
        )
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, E) -> (B, C, E)"""
        # Residual connection.
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class DecoderTransformer(nn.Module):

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_size: int,
        head_size: int,
        context_length: int,
        num_heads: int,
        ff_hidden_scaler: int,
        dropout_rate: float,
        num_layers: int,
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(context_length, embed_size)
        self.attention_layers = nn.Sequential(
            *[
                AttentionLayer(
                    embed_size=embed_size,
                    head_size=head_size,
                    context_length=context_length,
                    dropout_rate=dropout_rate,
                    num_heads=num_heads,
                    ff_hidden_scaler=ff_hidden_scaler,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

        # context_length is needed during inference.
        self.register_buffer("context_length", torch.tensor(context_length))

    def forward(self, x, targets=None):
        """Conducts training if targets not specified, otherwise inference.
        Args:
            x is batch of arrays of token indexes. (B, C).
            targets: batch of token indexes. (B).
        """
        batch_size, context_length = x.shape

        token_embed = self.token_embedding_table(x)  # (B, C, E)
        position_embed = self.position_embedding_table(  # (B, C, E)
            torch.arange(context_length, device=next(self.parameters()).device.type)
        )
        x = token_embed + position_embed  # (B, C, E) + (B, C, E) -> (B, C, E)
        x = self.attention_layers(x)  # (B, C, E) -> (B, C, E)
        x = self.layer_norm(x)  # (B, C, E) -> (B, C, E)
        logits = self.linear(x)  # (B, C, V)

        loss = None
        if targets is not None:
            batch_size, context_length, vocab_size = logits.shape
            logits = logits.view(batch_size * context_length, vocab_size)
            targets = targets.view(batch_size * context_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # TODO: consider making num_params a mixin for all nn.Module subclasses.
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
