import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from utils.story_pipeline import process_story

from utils.model_factory import build_model_and_dataset

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Interactive GPT Laboratory",
    page_icon="🧠",
    layout="wide"
)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "story_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
    "story_result": None,
    "components": None,
    "trained": False,
    "training_history": [],
    "last_generated": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def require_model():
    return (
        st.session_state.get("components") is not None
        and st.session_state.get("story_result") is not None
    )


def plot_line_chart(history, x_key, y_keys, labels, title, y_title):
    fig = go.Figure()

    for key, label in zip(y_keys, labels):
        fig.add_trace(
            go.Scatter(
                x=[row["epoch"] for row in history],
                y=[row[key] for row in history],
                mode="lines+markers",
                name=label
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_title,
        height=360,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig


def tensor_summary(tensor, name):
    tensor = tensor.detach().float().cpu()
    flat = tensor.flatten()

    return {
        "Tensor": name,
        "Shape": str(tuple(tensor.shape)),
        "Elements": tensor.numel(),
        "Minimum": float(flat.min()) if flat.numel() else 0.0,
        "Maximum": float(flat.max()) if flat.numel() else 0.0,
        "Mean": float(flat.mean()) if flat.numel() else 0.0,
        "Standard deviation": float(flat.std(unbiased=False)) if flat.numel() else 0.0
    }


def show_heatmap(matrix, title, x_labels=None, y_labels=None,
                 height=420, zmin=None, zmax=None):
    values = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else matrix

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value")
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Feature / Key position",
        yaxis_title="Token / Query position",
        margin=dict(l=20, r=20, t=55, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def save_story_checkpoint(model, tokenizer, model_components):
    payload = {
        "model_state_dict": model.state_dict(),
        "tokenizer_stoi": tokenizer.stoi,
        "tokenizer_itos": tokenizer.itos,
        "vocab_size": tokenizer.vocab_size,
        "block_size": model_components["block_size"],
        "embedding_dim": config.n_embd,
        "num_heads": config.n_head,
        "num_layers": config.n_layer
    }
    torch.save(payload, CHECKPOINT_PATH)


def generate_with_trace(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    greedy
):
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("The prompt did not produce any tokens.")

    generated_ids = list(prompt_ids)
    trace = []

    eos_id = tokenizer.stoi.get("<EOS>")
    pad_id = tokenizer.stoi.get("<PAD>")

    with torch.no_grad():
        for step in range(max_new_tokens):
            context = generated_ids[-model.block_size:]
            input_ids = torch.tensor(
                [context],
                dtype=torch.long,
                device=DEVICE
            )

            logits, _ = model(input_ids)
            next_logits = logits[0, -1, :] / max(temperature, 1e-5)

            if pad_id is not None:
                next_logits[pad_id] = float("-inf")

            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.numel())
                top_values, _ = torch.topk(next_logits, k)
                threshold = top_values[-1]
                next_logits[next_logits < threshold] = float("-inf")

            probabilities = torch.softmax(next_logits, dim=-1)

            top_probs, top_ids = torch.topk(
                probabilities,
                k=min(5, probabilities.numel())
            )

            if greedy:
                next_id = int(torch.argmax(probabilities).item())
            else:
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())

            chosen_probability = float(probabilities[next_id].item())

            trace.append({
                "Step": step + 1,
                "Context": tokenizer.decode(context),
                "Top candidates": ", ".join(
                    f"{tokenizer.itos[token_id]} ({prob:.1%})"
                    for token_id, prob in zip(
                        top_ids.tolist(), top_probs.tolist()
                    )
                ),
                "Selected token": tokenizer.itos[next_id],
                "Selected probability": chosen_probability
            })

            generated_ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

    return tokenizer.decode(generated_ids), trace


# ==================================================
# HEADER AND SIDEBAR
# ==================================================

st.title("🧠 Interactive GPT Laboratory")
st.caption(
    "A hands-on laboratory for understanding how a decoder-only "
    "Transformer learns and generates text."
)

with st.sidebar:
    st.header("Model Configuration")
    st.metric("Embedding dimension", config.n_embd)
    st.metric("Attention heads", config.n_head)
    st.metric("Transformer blocks", config.n_layer)
    st.metric("Maximum context", config.block_size)
    st.metric("Compute device", str(DEVICE))

    st.divider()
    st.caption(
        "This is a small GPT trained from scratch on the story you provide. "
        "It is an educational model, not a pretrained language model."
    )

if require_model():
    status = "Trained" if st.session_state["trained"] else "Created — not trained"
    st.info(f"Current story model: {status}")
else:
    st.info("Start by submitting a story in the Story & Data tab.")


# ==================================================
# MAIN TABS
# ==================================================

story_tab, transformer_tab, training_tab, generation_tab, theory_tab = st.tabs(
    [
        "📖 Story & Data",
        "🔬 Transformer Inspector",
        "📈 Training Studio",
        "✍️ Generation Studio",
        "🧠 How GPT Works"
    ]
)


# ==================================================
# TAB 1: STORY INPUT AND DATA PIPELINE
# ==================================================

with story_tab:
    st.header("1. Give GPT a story")

    story_text = st.text_area(
        "Enter up to 50 sentences",
        height=220,
        placeholder=(
            "The little cat sat beside the window. "
            "It watched birds flying across the garden."
        ),
        key="story_text"
    )

    if st.button(
        "Process story and build a fresh GPT",
        type="primary",
        use_container_width=True
    ):
        st.session_state["story_result"] = None
        st.session_state["components"] = None
        st.session_state["trained"] = False
        st.session_state["training_history"] = []
        st.session_state["last_generated"] = None

        with st.spinner("Validating, tokenizing and constructing the model..."):
            try:
                result = process_story(story_text)

                if not result["success"]:
                    st.error(result["message"])
                else:
                    components = build_model_and_dataset(
                        tokenizer=result["tokenizer"],
                        token_ids=result["token_ids"]
                    )

                    components["model"] = components["model"].to(DEVICE)

                    st.session_state["story_result"] = result
                    st.session_state["components"] = components

                    st.success(
                        "Story processed. A new GPT model has been initialized."
                    )

            except Exception as error:
                st.error(f"Could not process this story: {error}")

    if require_model():
        result = st.session_state["story_result"]
        components = st.session_state["components"]
        tokenizer = result["tokenizer"]
        token_ids = result["token_ids"]
        dataset = components["dataset"]

        st.divider()
        st.subheader("Story statistics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentences", result["sentence_count"])
        c2.metric("Vocabulary", components["vocab_size"])
        c3.metric("Encoded tokens", len(token_ids))
        c4.metric("Training windows", components["dataset_size"])

        st.subheader("A. Tokenization")

        token_rows = []
        for position, token_id in enumerate(token_ids):
            token_rows.append({
                "Position": position,
                "Token": tokenizer.itos[token_id],
                "Token ID": token_id,
                "Special token": tokenizer.itos[token_id] in tokenizer.special_tokens
            })

        st.dataframe(
            pd.DataFrame(token_rows),
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Vocabulary mapping"):
            vocab_rows = [
                {"Token": token, "ID": token_id}
                for token, token_id in tokenizer.stoi.items()
            ]
            st.dataframe(
                pd.DataFrame(vocab_rows),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("B. Encoded story")
        st.code(str(token_ids))

        st.subheader("C. Sliding-window dataset")

        selected_window = st.slider(
            "Choose a training example",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            key="dataset_window"
        )

        x, y = dataset[selected_window]
        x_ids = x.tolist()
        y_ids = y.tolist()

        left, right = st.columns(2)
        with left:
            st.markdown("**Input X**")
            st.code(str(x_ids))
            st.info(tokenizer.decode(x_ids))
        with right:
            st.markdown("**Target Y (shifted by one token)**")
            st.code(str(y_ids))
            st.info(tokenizer.decode(y_ids))

        st.caption(
            "At each position, the model receives the token in X and learns "
            "to predict the corresponding next token in Y."
        )

        st.subheader("D. Model dimensions")
        model = components["model"]
        st.write({
            "Input batch shape": (1, components["block_size"]),
            "Embedding dimension": config.n_embd,
            "Vocabulary size": components["vocab_size"],
            "Output logits shape": (
                1,
                components["block_size"],
                components["vocab_size"]
            ),
            "Trainable parameters": components["trainable_parameters"]
        })


# ==================================================
# TAB 2: TRANSFORMER INTERNAL INSPECTOR
# ==================================================

with transformer_tab:
    st.header("2. Inspect what happens inside GPT")

    if not require_model():
        st.warning("Process a story first.")
    else:
        result = st.session_state["story_result"]
        components = st.session_state["components"]
        model = components["model"]
        tokenizer = result["tokenizer"]
        dataset = components["dataset"]

        max_index = len(dataset) - 1
        sample_index = st.slider(
            "Select dataset window",
            min_value=0,
            max_value=max_index,
            value=0,
            key="inspector_window"
        )

        sample_x, _ = dataset[sample_index]
        input_ids = sample_x.unsqueeze(0).to(DEVICE)
        token_labels = [
            tokenizer.itos[token_id]
            for token_id in sample_x.tolist()
        ]

        model.eval()
        with torch.no_grad():
            logits, attention_weights, debug = model(
                input_ids,
                return_debug=True
            )

        st.success(
            "Inspector is displaying tensors from the actual current model."
        )

        # Embedding stage
        st.subheader("A. Token and positional embeddings")

        embedding_debug = debug["embeddings"]
        token_vectors = embedding_debug["token_vectors"][0]
        position_vectors = embedding_debug["position_vectors"]
        combined_vectors = embedding_debug["combined_embeddings"][0]

        embedding_metrics = pd.DataFrame([
            tensor_summary(token_vectors, "Token embeddings"),
            tensor_summary(position_vectors, "Position embeddings"),
            tensor_summary(combined_vectors, "Combined embeddings")
        ])
        st.dataframe(embedding_metrics, use_container_width=True, hide_index=True)

        embedding_choice = st.selectbox(
            "Choose embedding representation",
            [
                "Token embeddings",
                "Positional embeddings",
                "Combined embeddings"
            ]
        )

        embedding_map = {
            "Token embeddings": token_vectors,
            "Positional embeddings": position_vectors,
            "Combined embeddings": combined_vectors
        }

        show_heatmap(
            embedding_map[embedding_choice],
            embedding_choice,
            x_labels=[f"D{i}" for i in range(config.n_embd)],
            y_labels=token_labels,
            height=480
        )

        st.divider()
        st.subheader("B. Transformer block computations")

        block_debug_list = debug["blocks"]
        block_index = st.selectbox(
            "Select Transformer block",
            list(range(len(block_debug_list))),
            format_func=lambda index: f"Block {index + 1}",
            key="debug_block"
        )

        selected_block_debug = block_debug_list[block_index]
        attention_debug = selected_block_debug["attention"]
        head_debug_list = attention_debug["heads"]

        head_index = st.selectbox(
            "Select attention head",
            list(range(len(head_debug_list))),
            format_func=lambda index: f"Head {index + 1}",
            key="debug_head"
        )

        head_debug = head_debug_list[head_index]

        tensor_rows = [
            tensor_summary(head_debug["query"], "Query (Q)"),
            tensor_summary(head_debug["key"], "Key (K)"),
            tensor_summary(head_debug["value"], "Value (V)"),
            tensor_summary(head_debug["raw_scores"], "Raw attention scores"),
            tensor_summary(head_debug["masked_scores"], "Masked scores"),
            tensor_summary(head_debug["attention_weights"], "Attention weights"),
            tensor_summary(head_debug["weighted_values"], "Weighted values")
        ]

        st.markdown("**Actual tensor shapes and statistics**")
        st.dataframe(
            pd.DataFrame(tensor_rows),
            use_container_width=True,
            hide_index=True
        )

        q_col, k_col, v_col = st.columns(3)
        q_col.metric("Q shape", str(tuple(head_debug["query"].shape)))
        k_col.metric("K shape", str(tuple(head_debug["key"].shape)))
        v_col.metric("V shape", str(tuple(head_debug["value"].shape)))

        st.markdown("**Query, Key and Value vectors**")
        qkv_choice = st.selectbox(
            "Select tensor",
            ["Query (Q)", "Key (K)", "Value (V)"]
        )
        qkv_tensor = {
            "Query (Q)": head_debug["query"][0],
            "Key (K)": head_debug["key"][0],
            "Value (V)": head_debug["value"][0]
        }[qkv_choice]

        show_heatmap(
            qkv_tensor,
            f"{qkv_choice} — Head {head_index + 1}",
            x_labels=[f"head-dim-{i}" for i in range(qkv_tensor.shape[-1])],
            y_labels=token_labels
        )

        st.markdown("**Attention score pipeline**")
        score_choice = st.selectbox(
            "Choose attention computation",
            [
                "Raw scores (QKᵀ / √d)",
                "Causal mask",
                "Masked scores",
                "Softmax attention weights"
            ]
        )

        if score_choice == "Raw scores (QKᵀ / √d)":
            score_tensor = head_debug["raw_scores"][0]
        elif score_choice == "Causal mask":
            score_tensor = head_debug["causal_mask"]
        elif score_choice == "Masked scores":
            score_tensor = head_debug["masked_scores"][0]
        else:
            score_tensor = head_debug["attention_weights"][0]

        if score_choice == "Masked scores":
            display_tensor = torch.nan_to_num(
                score_tensor,
                nan=0.0,
                posinf=0.0,
                neginf=-10.0
            )
        else:
            display_tensor = score_tensor

        show_heatmap(
            display_tensor,
            score_choice,
            x_labels=token_labels,
            y_labels=token_labels,
            zmin=0 if (
                "weights" in score_choice.lower()
                or score_choice == "Causal mask"
            ) else None,
            zmax=1 if (
                "weights" in score_choice.lower()
                or score_choice == "Causal mask"
            ) else None
        )

        st.caption(
            "Rows are query positions. Columns are key positions. "
            "The causal mask prevents a position from attending to future tokens."
        )

        st.divider()
        st.subheader("C. Multi-head attention output")

        multi_head = attention_debug
        st.write({
            "Stacked head outputs": tuple(multi_head["head_outputs"].shape),
            "Concatenated heads": tuple(multi_head["concatenated_heads"].shape),
            "Output projection": tuple(multi_head["output_projection"].shape)
        })

        st.divider()
        st.subheader("D. Feed-forward network")

        mlp_debug = selected_block_debug["mlp"]
        mlp_choice = st.selectbox(
            "Select MLP stage",
            [
                "Linear expansion",
                "GELU activation",
                "Linear projection"
            ]
        )

        mlp_map = {
            "Linear expansion": mlp_debug["linear_output"][0],
            "GELU activation": mlp_debug["gelu_output"][0],
            "Linear projection": mlp_debug["final_output"][0]
        }
        mlp_tensor = mlp_map[mlp_choice]

        st.write("Activation shape:", tuple(mlp_tensor.shape))
        show_heatmap(
            mlp_tensor,
            mlp_choice,
            x_labels=[f"feature-{i}" for i in range(mlp_tensor.shape[-1])],
            y_labels=token_labels,
            height=500
        )

        st.divider()
        st.subheader("E. Residual connections and normalization")

        residual_rows = [
            tensor_summary(
                selected_block_debug["residual_input"],
                "Block input"
            ),
            tensor_summary(
                selected_block_debug["normalized_for_attention"],
                "After LayerNorm 1"
            ),
            tensor_summary(
                selected_block_debug["attention_output"],
                "Attention output"
            ),
            tensor_summary(
                selected_block_debug["after_attention_residual"],
                "After attention residual"
            ),
            tensor_summary(
                selected_block_debug["normalized_for_mlp"],
                "After LayerNorm 2"
            ),
            tensor_summary(
                selected_block_debug["mlp"]["final_output"],
                "MLP output"
            ),
            tensor_summary(
                selected_block_debug["block_output"],
                "Block output"
            )
        ]

        st.dataframe(
            pd.DataFrame(residual_rows),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("F. Final logits and probabilities")

        final_logits = logits[0, -1]
        final_probabilities = torch.softmax(final_logits, dim=-1)

        top_probs, top_ids = torch.topk(
            final_probabilities,
            k=min(10, final_probabilities.numel())
        )

        output_df = pd.DataFrame([
            {
                "Token": tokenizer.itos[token_id],
                "Logit": float(final_logits[token_id].item()),
                "Probability": float(probability.item())
            }
            for token_id, probability in zip(
                top_ids.tolist(), top_probs
            )
        ])

        st.dataframe(
            output_df.style.format({
                "Logit": "{:.4f}",
                "Probability": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        probability_fig = go.Figure(
            go.Bar(
                x=output_df["Token"],
                y=output_df["Probability"]
            )
        )
        probability_fig.update_layout(
            title="Next-token probability distribution at the final position",
            yaxis_tickformat=".0%",
            height=350
        )
        st.plotly_chart(probability_fig, use_container_width=True)

        with st.expander("Inspect raw logits tensor"):
            st.write(debug["logits"].detach().cpu())


# ==================================================
# TAB 3: LIVE TRAINING STUDIO
# ==================================================

with training_tab:
    st.header("3. Train your GPT")

    if not require_model():
        st.warning("Process a story first.")
    else:
        components = st.session_state["components"]
        model = components["model"]
        dataset = components["dataset"]

        st.write(
            "Training updates the parameters of the current story-specific GPT. "
            "The charts below are updated from real training measurements."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=100,
                step=25
            )

        with col2:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=128,
                value=min(8, len(dataset)),
                step=1
            )

        with col3:
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.00001,
                max_value=0.1,
                value=0.001,
                format="%.5f"
            )

        if st.session_state["trained"]:
            st.success("This story model has already been trained.")
            if st.button("Reset model to fresh random initialization"):
                result = st.session_state["story_result"]
                fresh_components = build_model_and_dataset(
                    result["tokenizer"],
                    result["token_ids"]
                )
                fresh_components["model"] = fresh_components["model"].to(DEVICE)
                st.session_state["components"] = fresh_components
                st.session_state["trained"] = False
                st.session_state["training_history"] = []
                st.rerun()

        train_clicked = st.button(
            "🚀 Start training",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["trained"]
        )

        if train_clicked:
            model = components["model"]
            model.train()

            loader = DataLoader(
                dataset,
                batch_size=min(int(batch_size), len(dataset)),
                shuffle=True
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(learning_rate)
            )
            criterion = nn.CrossEntropyLoss()

            history = []
            progress = st.progress(0)
            status = st.empty()
            metric_area = st.empty()
            chart_area = st.empty()

            initial_parameter_norm = math.sqrt(
                sum(
                    float(parameter.detach().pow(2).sum().item())
                    for parameter in model.parameters()
                )
            )

            for epoch in range(1, int(epochs) + 1):
                epoch_losses = []
                epoch_gradient_norms = []

                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)

                    optimizer.zero_grad(set_to_none=True)

                    batch_logits, _ = model(batch_x)

                    loss = criterion(
                        batch_logits.reshape(-1, batch_logits.size(-1)),
                        batch_y.reshape(-1)
                    )

                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    epoch_gradient_norms.append(float(grad_norm.item()))

                    optimizer.step()
                    epoch_losses.append(float(loss.item()))

                average_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                average_grad_norm = (
                    sum(epoch_gradient_norms) /
                    max(len(epoch_gradient_norms), 1)
                )

                parameter_norm = math.sqrt(
                    sum(
                        float(parameter.detach().pow(2).sum().item())
                        for parameter in model.parameters()
                    )
                )

                history.append({
                    "epoch": epoch,
                    "loss": average_loss,
                    "gradient_norm": average_grad_norm,
                    "parameter_norm": parameter_norm
                })

                progress.progress(epoch / int(epochs))
                status.write(
                    f"Epoch {epoch}/{int(epochs)} — "
                    f"loss: {average_loss:.5f}"
                )

                metric_area.metric(
                    "Current training loss",
                    f"{average_loss:.5f}",
                    delta=None
                )

                chart_area.plotly_chart(
                    plot_line_chart(
                        history,
                        "epoch",
                        ["loss"],
                        ["Training loss"],
                        "Training loss by epoch",
                        "Cross-entropy loss"
                    ),
                    use_container_width=True,
                    key=f"live_loss_{epoch}"
                )

            model.eval()

            st.session_state["training_history"] = history
            st.session_state["trained"] = True

            tokenizer = st.session_state["story_result"]["tokenizer"]
            save_story_checkpoint(model, tokenizer, components)

            final_parameter_norm = history[-1]["parameter_norm"]

            st.success(
                f"Training complete. Model checkpoint saved to {CHECKPOINT_PATH}."
            )

            st.write({
                "Initial parameter L2 norm": initial_parameter_norm,
                "Final parameter L2 norm": final_parameter_norm,
                "Final training loss": history[-1]["loss"],
                "Epochs completed": int(epochs)
            })

            st.rerun()

        history = st.session_state["training_history"]

        if history:
            st.subheader("Training history")

            st.plotly_chart(
                plot_line_chart(
                    history,
                    "epoch",
                    ["loss"],
                    ["Training loss"],
                    "Training loss",
                    "Cross-entropy loss"
                ),
                use_container_width=True
            )

            st.plotly_chart(
                plot_line_chart(
                    history,
                    "epoch",
                    ["gradient_norm", "parameter_norm"],
                    ["Gradient norm", "Parameter norm"],
                    "Optimization diagnostics",
                    "Magnitude"
                ),
                use_container_width=True
            )

            st.dataframe(
                pd.DataFrame(history),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These are training metrics, not validation metrics. "
                "A tiny story can be memorized; low training loss does not "
                "demonstrate general language understanding."
            )


# ==================================================
# TAB 4: GENERATION STUDIO
# ==================================================

with generation_tab:
    st.header("4. Generate text from your story model")

    if not require_model():
        st.warning("Process a story first.")
    elif not st.session_state["trained"]:
        st.warning("Train the story-specific model first in Training Studio.")
    else:
        components = st.session_state["components"]
        model = components["model"]
        tokenizer = st.session_state["story_result"]["tokenizer"]

        prompt = st.text_input(
            "Generation prompt",
            placeholder="Enter a beginning from your story...",
            key="generation_prompt"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            max_new_tokens = st.slider(
                "New tokens",
                min_value=1,
                max_value=100,
                value=20
            )

        with c2:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1
            )

        with c3:
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=min(20, tokenizer.vocab_size),
                value=min(5, tokenizer.vocab_size)
            )

        greedy = st.checkbox("Greedy decoding (always choose highest probability)")

        if st.button(
            "Generate token by token",
            type="primary",
            use_container_width=True
        ):
            if not prompt.strip():
                st.warning("Enter a prompt first.")
            else:
                try:
                    generated_text, trace = generate_with_trace(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        greedy=greedy
                    )

                    st.session_state["last_generated"] = {
                        "text": generated_text,
                        "trace": trace
                    }

                except Exception as error:
                    st.error(f"Generation failed: {error}")

        generated = st.session_state.get("last_generated")

        if generated:
            st.subheader("Generated continuation")
            st.success(generated["text"])

            st.subheader("Generation trace")

            trace_df = pd.DataFrame(generated["trace"])
            st.dataframe(
                trace_df,
                use_container_width=True,
                hide_index=True
            )

            if not trace_df.empty:
                st.bar_chart(
                    trace_df.set_index("Step")["Selected probability"]
                )

            st.caption(
                "Each row records the context, top candidates, selected token "
                "and probability assigned to that selection."
            )


# ==================================================
# TAB 5: CONCEPTUAL LEARNING GUIDE
# ==================================================

with theory_tab:
    st.header("5. Follow the data through GPT")

    stages = {
        "Tokenization": (
            "Raw text is split into tokens. The tokenizer maps each token "
            "to an integer ID from the vocabulary."
        ),
        "Embeddings": (
            "Each token ID selects a learned vector. A learned positional "
            "vector is added so the model can distinguish sequence positions."
        ),
        "Attention": (
            "Queries and keys produce compatibility scores. The causal mask "
            "blocks future positions. Softmax turns the allowed scores into "
            "weights, which combine value vectors."
        ),
        "Feed-forward network": (
            "Each token position passes independently through a learned "
            "expansion, GELU nonlinearity and projection back to the model width."
        ),
        "Residual connections": (
            "The sublayer output is added to its input. This provides a direct "
            "path for information and gradients."
        ),
        "Training": (
            "The model produces logits at every sequence position. Cross-entropy "
            "compares them with the shifted target IDs. Backpropagation computes "
            "gradients and the optimizer updates parameters."
        ),
        "Generation": (
            "The final-position logits become a probability distribution. "
            "A decoding rule selects one next token, which is appended to the "
            "context before the next prediction."
        )
    }

    selected = st.selectbox("Choose a stage", list(stages.keys()))
    st.info(stages[selected])

    st.markdown("### The complete data flow")
    st.code(
        """
Story text
   |
   v
Tokenizer -> Vocabulary -> Token IDs
   |
   v
Sliding windows -> (Input X, Target Y)
   |
   v
Token embeddings + Positional embeddings
   |
   v
Transformer block 1
   |-- LayerNorm
   |-- Causal multi-head self-attention
   |-- Residual addition
   |-- LayerNorm
   |-- Feed-forward network
   |-- Residual addition
   |
   v
More Transformer blocks
   |
   v
Final LayerNorm -> Linear language head -> Logits
   |
   v
Cross-entropy loss -> Backpropagation -> Optimizer update
   |
   v
Next-token probabilities -> Sampling -> Generated token
        """.strip(),
        language="text"
    )
