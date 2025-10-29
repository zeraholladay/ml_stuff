# ml_stuff

A notebook of ML concepts.

## Venv setup

```bash
python3 -mvenv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Tokenization

This repo contains a minimal example at `tokenizer/simple_hugging_face.py` that demonstrates text tokenization using Hugging Face Transformers.

### What is tokenization?
- **Tokenization** converts raw text into integer IDs that models can process.
- Modern LLMs use **subword tokenizers** (e.g., WordPiece/BPE), breaking words into pieces to balance vocabulary size and coverage (e.g., "unbelievable" → "un", "##bel", "##ievable").

### Key concepts shown in `simple_hugging_face.py`
- **AutoTokenizer.from_pretrained("bert-base-uncased")**: loads the matching tokenizer config, vocabulary, and rules (lowercasing for uncased models).
- **Encoding**: `tok(input_text, return_tensors="pt")` returns a dictionary (PyTorch tensors) including:
  - `input_ids`: token IDs after subword tokenization (with special tokens added).
  - `attention_mask`: 1 where tokens are real, 0 where padding is present.
  - `token_type_ids` (for some models): segment IDs (e.g., sentence A/B in BERT).
- **Special tokens**: models may add tokens like `[CLS]` (start) and `[SEP]` (separator) automatically.
- **Padding & truncation** (not enabled by default here):
  - `padding=True` pads shorter sequences to the same length.
  - `truncation=True` clips longer texts to the model’s max length.
- **Decoding**: you can map IDs back to text via `tokenizer.decode(input_ids)` (useful for inspection).

### How to run the example
- Run: `python tokenizer/simple_hugging_face.py`
- It prints the original text and the tokenized outputs (PyTorch tensors).


## Linear Regression: The Simplest Neural Network

This class of problems is defined by the goal of predicting a continuous output value based on one or more input variables.

This repository includes a minimal PyTorch example at `neural_nets/simple_add_n.py` that demonstrates linear regression as the simplest form of a neural network.

![Example](./docs/img/Okuns_law_quarterly_differences.svg.png)

### Simplest Form of Neural Network
- A linear regression model, whether built in PyTorch or elsewhere, is structurally the simplest form of a neural network. It consists of a single layer with no activation function (or a linear/identity activation function).

### Core Components
- **Weights (W) and Biases (b)**: These are the learnable parameters the model optimizes during training.
- **Linear Transformation**: The core operation is the weighted sum of inputs, expressed as \(\hat{y} = W x + b\).

### PyTorch Implementation
- In PyTorch, a linear model is typically defined using `torch.nn.Linear(input_size, output_size)`.
- The `nn.Linear` class is fundamental to building many neural networks in PyTorch. A simple linear regression model is just a specific, minimal instance of the `nn.Module` architecture.

```python
import torch
import torch.nn as nn

# One-dimensional linear regression: y_hat = w * x + b
model = nn.Sequential(nn.Linear(1, 1))

# Example forward pass with a batch of inputs shaped [N, 1]
x = torch.tensor([[0.0], [1.0], [2.0]])
y_hat = model(x)
```

### Example in This Repo
See `neural_nets/simple_add_n.py` for a complete, commented example that:
  - Creates simple training data,
  - Trains a 1D linear layer with Mean Squared Error (MSE) loss and the Adam optimizer,
  - Evaluates on a test domain and visualizes the results.

The model is trained constant-offset mapping of the form y = x + 2. It is then tested for cos(x) target vs prediction cos(x) + 2 over x ∈ [0, 2π].

![Linear Regression](./docs/img/simple_add_n.png)
