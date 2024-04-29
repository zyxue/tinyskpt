import dataclasses
import itertools
from pathlib import Path
from typing import Callable

import altair as alt
import model_output_config
import pandas as pd
import streamlit as st
import torch
from torch.nn import functional as F

from tinyskpt import token_utils

st.set_page_config(layout="wide")

st.header("Tiny GPT playground")


_DEVICE = model_output_config.DEVICE


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

raw_input = st.text_input(label="Enter a prompt")
if not raw_input:
    st.stop()


hook_outputs = {}


def getActivation(name: str) -> Callable:
    global hook_outputs

    def hook(model, input, output):
        hook_outputs[name] = output.detach()

    return hook


key_hooks_list, query_hooks_list = [], []
for layer_index, layer in enumerate(model.attention_layers):
    key_hooks, query_hooks = [], []
    for head_index, head in enumerate(layer.multi_head_attention.heads):
        key_hooks.append(
            head.key.register_forward_hook(
                getActivation(f"key_layer{layer_index}_head{head_index}")
            )
        )
        query_hooks.append(
            head.query.register_forward_hook(
                getActivation(f"query_layer{layer_index}_head{head_index}")
            )
        )
    key_hooks_list.append(key_hooks)
    query_hooks_list.append(query_hooks)


context = torch.tensor([tokenizer.encode(raw_input)], dtype=torch.long).to(_DEVICE)
logits, _ = model(context)
logits_slice = logits[0, -1, :]  # (B, C, V) => (V)
probs = F.softmax(logits_slice, dim=-1)  # (B, V) => (V)

df_probs = pd.DataFrame(
    {
        "index": list(range(tokenizer.vocab_size)),
        "next_token": list(tokenizer.decode(list(range(tokenizer.vocab_size)))),
        "prob": probs.to("cpu").detach().numpy(),
    }
)

col0, col1 = st.columns(spec=[0.3, 0.7])

with col0:
    st.altair_chart(
        alt.Chart(df_probs.sort_values("prob", ascending=False).head(10))
        .mark_bar()
        .encode(
            x=alt.X("next_token", sort=None),
            y=alt.Y("prob", scale=alt.Scale(domain=[0, 1])),
        )
    )

# Inspect attention heads
embed_size = model.token_embedding_table.embedding_dim
context_length = len(raw_input)

dfs_list = []  # Nested lists of dfs by layer.
for layer_index, (key_hooks, query_hooks) in enumerate(
    zip(key_hooks_list, query_hooks_list)
):
    dfs = []
    for head_index in range(len(key_hooks)):
        key = hook_outputs[f"key_layer{layer_index}_head{head_index}"]
        assert context_length == key.shape[1]
        query = hook_outputs[f"query_layer{layer_index}_head{head_index}"]

        key_transpose = key.transpose(dim0=1, dim1=2)  # (B, C, H) -> (B, H, C)
        weight_raw = query @ key_transpose  # (B, C, H) @ (B, H, C) -> (B, C, C)
        weight_normalized = weight_raw / embed_size**0.5
        tril = torch.tril(torch.ones(context_length, context_length).to(_DEVICE))
        weight_masked = weight_normalized.masked_fill(
            tril[:context_length, :context_length] == 0, float("-inf")
        )
        weight = F.softmax(weight_masked, dim=-1)
        df_weight = pd.DataFrame(
            weight.to("cpu").detach().numpy()[0],
            # Prefix with index to avoid duplicate column/index names.
            columns=[f"{i}:{t}" for i, t in enumerate(raw_input)],
            index=[f"{i}:{t}" for i, t in enumerate(raw_input)],
        )

        # Takes tail(1) as we only want the attentions wrt. the prediction of
        # the next token given input_raw.
        dfs.append(
            df_weight.tail(1).assign(
                layer_idx=layer_index,
                head_idx=head_index,
            )
        )
    dfs_list.append(dfs)

with col1:
    subcol0, subcol1 = st.columns(2)
    with subcol0:
        for idx, dfs in enumerate(dfs_list[: len(dfs_list) // 2]):
            st.write(f"Attention layer {idx}")
            df_plot = (
                pd.concat(dfs)
                .rename(columns={"layer_idx": "Layer", "head_idx": "Head"})
                .drop(columns=["Layer"])
                .set_index(["Head"])
            )
            st.dataframe(
                df_plot.style.format(precision=2).background_gradient(
                    axis=None, vmin=0, vmax=1, cmap="YlGnBu"
                ),
            )
    with subcol1:
        # TODO: dedup code with above.
        for idx, dfs in enumerate(dfs_list[len(dfs_list) // 2 :]):
            st.write(f"Attention layer {idx}")
            df_plot = (
                pd.concat(dfs)
                .rename(columns={"layer_idx": "Layer", "head_idx": "Head"})
                .drop(columns=["Layer"])
                .set_index(["Head"])
            )
            st.dataframe(
                df_plot.style.format(precision=2).background_gradient(
                    axis=None, vmin=0, vmax=1, cmap="YlGnBu"
                ),
            )

with col0:
    st.altair_chart(
        alt.Chart(
            pd.concat(itertools.chain(*dfs_list))
            .set_index(["layer_idx", "head_idx"])
            .stack()
            .rename_axis(index={None: "token"})
            .to_frame("weight")
            .reset_index()
        )
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.X("weight:Q").bin(extent=[0, 1], step=0.025),
            alt.Y("count()").stack(None),
            alt.Color("token:N"),
        )
    )
