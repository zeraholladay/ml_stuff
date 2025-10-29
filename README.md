# ml_stuff

A notebook of ML concepts.

### Setup:

```bash
python3 -mvenv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Tensors

Tensors are the **fundamental data structure** used in modern machine learning, particularly in frameworks like PyTorch and TensorFlow.

Simply put, a tensor is a **container** that holds data, almost always **numerical data**, and is typically a generalization of matrices to an arbitrary number of dimensions.

## Tensors as Data Containers

You can think of a tensor as a multi-dimensional array or an N-dimensional grid of numbers. The "rank" or "order" of the tensor refers to the number of dimensions it has.

Here's how common data structures relate to tensors:

| Rank (Order) | Name | Description | Example |
| :---: | :--- | :--- | :--- |
| **0** | **Scalar** | A single number. | Temperature ($25$) |
| **1** | **Vector** | A list of numbers (a 1D array). | Coordinates ($[3, 5, 8]$) |
| **2** | **Matrix** | A rectangular grid of numbers (a 2D array). | A grayscale image (Height $\times$ Width) |
| **3** | **3D Tensor** | A cube or stack of matrices. | A color image (Height $\times$ Width $\times$ Color Channels) |
| **4+** | **N-D Tensor** | Used for batches of data or sequential data. | A batch of color images (Batch Size $\times$ Height $\times$ Width $\times$ Channels) |

## Why Tensors are Essential in ML

Tensors are more than just data storage; they are the primary medium for **mathematical operations** and **gradient computation** in neural networks.

1.  **Uniform Representation:** Tensors allow data of all types—from raw input (like text, images, or audio) to weights and biases within a model, to the output predictions—to be represented in a **consistent mathematical structure**.
2.  **GPU Acceleration:** Tensors are designed to be efficiently stored in contiguous memory blocks, making them highly suitable for processing by **GPUs** (Graphics Processing Units). GPUs are optimized for the massive parallel calculations (like matrix multiplications) required to train large neural networks.
3.  **Automatic Differentiation:** In frameworks like PyTorch, tensors are linked to a computational graph. This enables **automatic differentiation** (or Autograd), which is the process of automatically calculating the gradients required for the backpropagation algorithm during model training. Every tensor tracks its history of operations, allowing the framework to determine how to adjust the model's weights to minimize error. 

In summary, a tensor is the universal, multi-dimensional numeric data structure that makes the training of deep learning models mathematically consistent and computationally efficient.

## Tokenization

This repo contains a minimal example at `tokenizer/simple_hugging_face.py` that demonstrates text tokenization using Hugging Face Transformers.

See [https://tiktokenizer.vercel.app/](Tiktokenizer) too.

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
