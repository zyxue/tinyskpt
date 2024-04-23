"""Configuration related classes."""

import pydantic


class ArchConfig(pydantic.BaseModel):
    """Configurations related to the model architecture."""

    # Size of the vocabulary.
    vocab_size: int
    # Number of training examples per batch.
    batch_size: int
    # Size of the embedding vectors.
    embed_size: int
    # Number of attention heads.
    num_heads: int
    # Number of tokens in each context.
    context_length: int
    # The scaler for calculating the hidden layer size in the feedforward sublayer.
    ff_hidden_scaler: int
    # Number of attention layers. An attention layer has two sublayers: a
    # multi-head attention sublayer & a feedforward layer.
    num_layers: int
    # Dropout rate.
    dropout_rate: float

    @property
    def head_size(self) -> int:
        """Size of the attention head

        Aka. the dimension of key, query, or value. It's derived from the
        embed_size and num_heads.
        """
        return self.embed_size // self.num_heads


class TrainConfig(pydantic.BaseModel):
    """Configurations related to model training."""

    # The maximum number of iterations to run the training loop.
    max_iters: int
    # Learning rate.
    learning_rate: float


class EvalConfig(pydantic.BaseModel):
    """Configurations related to model evaluation.

    Evaluation is aka. validation.
    """

    # How often to evaluate the loss on the train and validation sets.
    eval_interval: int
    # How many batches to average over when estimating the loss.
    num_batches_for_eval: int
    # Fraction of data to use for validation, between 0 and 1.
    eval_size: float


class Config(pydantic.BaseModel):
    """Overall configuration object."""

    arch_config: ArchConfig
    train_config: TrainConfig
    eval_config: EvalConfig
