import model_output_config
import streamlit as st
import torch
import torch.nn.functional as F

from tinyskpt import attn, token_utils


def make_inference(
    model: attn.DecoderTransformer, context: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """Makes inference.

    Args:
        model: The model to use for inference.
        context: The context to start from. It could be just torch.zeros((1, 1),
            dtype=torch.long), i.e. start from nothing.
        max_new_tokens: The maximum number of tokens to generate.
    """
    out = context
    progress_bar = st.progress(0, "Predicting")

    for index in range(max_new_tokens):
        # Gets at max the last `context_length` tokens
        out = out[:, -model.context_length :]
        logits, _ = model(out)
        logits = logits[:, -1, :]  # (B, C, V) => (B, V)
        probs = F.softmax(logits, dim=-1)  # (B, V) => (B, V)
        new_token_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
        out = torch.cat((out, new_token_idx), dim=1)  # (B, T+1)

        progress_bar.progress(
            (index + 1) / max_new_tokens,
            text=f"Predicting {index + 1}/{max_new_tokens} tokens",
        )
    return out


model_output_key = st.selectbox(
    label="Select a model output",
    options=list(model_output_config.AVAILABLE_MODEL_OUTPUTS.keys()),
    index=None,
)

if not model_output_key:
    st.stop()

model_output = model_output_config.AVAILABLE_MODEL_OUTPUTS[model_output_key]
tokenizer = token_utils.Tokenizer.load(model_output.tokenizer_path)
model = model_output_config.load_model(model_output.model_path)

col0, col1 = st.columns(2)
with col0:
    num_batches = int(
        st.number_input(
            "How many batches to generate",
            min_value=1,
            max_value=8,
            value=3,
        )
    )
with col1:
    num_new_tokens = int(
        st.number_input(
            "How many tokens to generate",
            min_value=10,
            max_value=1000,
            value=500,
        ),
    )

context = torch.zeros(
    (num_batches, 1), dtype=torch.long, device=model_output_config.DEVICE
)

if st.button("Generate"):
    predictions = make_inference(model, context, num_new_tokens)

    for tokens in predictions.to("cpu").detach().numpy().tolist():
        st.code(tokenizer.decode(tokens))
