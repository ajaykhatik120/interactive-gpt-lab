# 🧠 Interactive GPT Laboratory

**Build, train, inspect, and understand a GPT-style Transformer from scratch.**

Interactive GPT Laboratory is an educational application built using **Python, PyTorch, and Streamlit**. It demonstrates how a GPT-style language model works internally, from tokenization and embeddings to causal self-attention, Transformer blocks, training, and text generation.

Instead of treating GPT as a black box, this project allows users to explore the actual computations happening inside the model.

## 🚀 Features

### 1. Story Builder
- Enter your own text or story.
- Build a vocabulary dynamically.
- Convert text into token IDs.
- Generate input-target training sequences.
- Inspect vocabulary and dataset dimensions.

### 2. Transformer Inspector
Explore the actual intermediate computations inside the Transformer:
- Token and positional embeddings.
- Query, Key, and Value matrices.
- Attention scores and causal masking.
- Softmax attention weights.
- Multi-head attention.
- Feed-forward network activations.
- Residual connections and Layer Normalization.
- Final logits and probability distributions.

### 3. Training Studio
- Train a GPT model from scratch.
- Monitor training loss.
- Inspect gradient and parameter norms.
- Save trained model checkpoints.

### 4. Generation Studio
- Generate text using your trained model.
- Observe token-by-token generation.
- Experiment with temperature and top-k sampling.
- Compare greedy decoding and probabilistic sampling.
- Inspect candidate token probabilities.

## 🏗️ Architecture

The project implements a decoder-only GPT-style Transformer.

```text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Token Embeddings
    +
Positional Embeddings
    |
    v
Transformer Blocks
    |
    +---- Causal Multi-Head Self-Attention
    |
    +---- Feed-Forward Network
    |
    +---- Residual Connections
    |
    +---- Layer Normalization
    |
    v
Final Layer Normalization
    |
    v
Linear Language Head
    |
    v
Logits
    |
    v
Softmax
    |
    v
Next Token Prediction
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Neural network implementation |
| Streamlit | Interactive web interface |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Plotly | Interactive visualizations |
| Matplotlib | Visualization |

## 📂 Project Structure

```text
interactive-gpt-lab/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── tokenizer/
│   └── word_tokenizer.py
│
├── dataset/
│   └── dataset.py
│
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── normalization.py
│   ├── transformer_block.py
│   └── gpt.py
│
├── training/
│   └── trainer.py
│
├── inference/
│   └── predictor.py
│
├── visualization/
│
├── utils/
│   ├── model_factory.py
│   └── story_pipeline.py
│
├── tests/
│
├── data/
└── checkpoints/
```

## ⚙️ Installation and Setup

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`.
- Git.
- Basic understanding of Python.

### 1. Clone the repository

```bash
git clone https://github.com/ajaykhatik120/interactive-gpt-lab.git
cd interactive-gpt-lab
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🧪 Run Tests

Execute the test suite:

```bash
python -m pytest tests/ -v
```

The tests cover the tokenizer, dataset, embeddings, attention, Transformer blocks, GPT architecture, training, and generation.

## 🧠 Learning Objectives

This project is designed to help understand:

1. How text is converted into numerical representations.
2. How embeddings represent tokens.
3. Why positional information is required.
4. How Query, Key, and Value projections work.
5. How causal self-attention prevents future-token access.
6. How multi-head attention learns different relationships.
7. Why feed-forward networks are used in Transformer blocks.
8. How residual connections and Layer Normalization help neural networks.
9. How next-token prediction is trained using cross-entropy loss.
10. How autoregressive text generation works.

## ⚠️ Current Limitations

- Uses a word-level tokenizer; BPE tokenization is a potential future extension.
- Designed primarily for educational experimentation.
- Small custom datasets can lead to memorization rather than generalization.
- It is not a replacement for large pretrained language models.
- Training performance depends on available CPU/GPU resources.

## 🔮 Future Improvements

- Implement Byte Pair Encoding (BPE).
- Add validation loss and evaluation metrics.
- Visualize embedding spaces.
- Add training checkpoints and model restoration.
- Experiment with different Transformer configurations.
- Add larger training datasets.
- Explore GPU acceleration and deployment improvements.

## 🎯 Project Goal

The goal of Interactive GPT Laboratory is to understand Generative AI by building the underlying architecture step by step, rather than simply using an existing pretrained model.

**Learn GPT by building GPT.**

## 👨‍💻 Author

**Ajay Kumar**

GitHub: [@ajaykhatik120](https://github.com/ajaykhatik120)

Project: [Interactive GPT Laboratory](https://github.com/ajaykhatik120/interactive-gpt-lab)

---

*Built with Python, PyTorch, and Streamlit.*