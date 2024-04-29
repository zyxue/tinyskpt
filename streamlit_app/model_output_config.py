import dataclasses
from pathlib import Path

import streamlit as st
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: factor to a more relevant util models.
@st.cache_resource
def load_model(path: str):
    return torch.load(path).to(DEVICE)


@dataclasses.dataclass(frozen=True)
class ModelOutput:
    key: str  # An identifier for the model output.
    output_dir: Path

    @property
    def tokenizer_path(self) -> Path:
        return self.output_dir / "tokenizer.json"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.pt"

    @property
    def loss_history_path(self) -> Path:
        return self.output_dir / "loss_history.json"


AVAILABLE_MODEL_OUTPUTS = {
    "tiny_shakespeare": ModelOutput(
        key="tiny_shakespeare",
        output_dir=Path(__file__).parent / "model_outputs/tinyshakespeare_output",
    ),
    "hongloumeng": ModelOutput(
        key="hongloumeng",
        output_dir=Path(__file__).parent / "model_outputs/hongloumeng_output",
    ),
}
