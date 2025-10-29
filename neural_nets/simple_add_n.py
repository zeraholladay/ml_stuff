#!/usr/bin/env python3
"""Minimal PyTorch example: learn an affine mapping y = x + c with a tiny network.

This script builds a single linear neuron (`nn.Linear(1, 1)`) and trains it with
mean squared error to recover a constant-offset mapping of the form y = x + c.
The toy dataset is produced by `make_input_and_output_features()`; with the current
settings it creates 100 points and an offset c = target_start - input_start.
The program runs a short training loop (Adam) and prints learned parameters,
loss, predictions, and a simple visualization.
"""

import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)

# Allow importing project root config when running this script directly
try:
    from config import device
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import device

offset = 2


def make_training_input_and_output_features():
    start = 0
    end = 100

    inputs = torch.arange(start, end, dtype=torch.float32).unsqueeze(1).to(device)
    outputs = (
        torch.arange(start + offset, end + offset, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    return inputs, outputs  # i.e. y = x + offset


def make_test_input_and_output_features():
    num_points: torch.Number = 100

    x = torch.linspace(
        0.0, 2.0 * torch.pi, steps=num_points, dtype=torch.float32, device=device
    )
    inputs = torch.cos(x).unsqueeze(1)
    outputs = (torch.cos(x) + offset).unsqueeze(1)

    return inputs, outputs, x.tolist()


def train_linear_model(inputs: torch.Tensor, targets: torch.Tensor) -> nn.Sequential:
    """Fit a 1D linear model y = w·x + b to map inputs → targets.

    A linear (affine) model transforms an input x by scaling it with a
    learnable weight w and then shifting by a learnable bias b: y_hat = w*x + b.
    With `nn.Linear(1, 1)` we have exactly two learnable parameters (w and b)
    that define this straight line. Training adjusts these parameters so that
    predictions y_hat match targets as closely as possible.
    """

    learning_rate = 0.1
    num_steps = 400

    # This layer implements the affine function y_hat = w*x + b with learnable
    # parameters w (weight) and b (bias). For in_features=1 and out_features=1,
    # it is the classic 1D linear regression model.
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Initialize near the expected solution to make convergence obvious
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()

    # Mean Squared Error (MSE) is the average of squared differences between
    # predictions and targets: MSE = mean((y_hat - y)^2). Squaring emphasizes
    # larger errors and yields a smooth objective for gradient descent.
    loss_fn = nn.MSELoss()

    # Optimizer: algorithm that updates parameters using gradients. Adam adapts
    # per-parameter learning rates based on running estimates of the first and
    # second moments of gradients (mean and uncentered variance). This often
    # converges faster and more robustly than plain SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(num_steps):
        # 1) Reset accumulated gradients from the previous step
        optimizer.zero_grad()

        # 2) Forward pass: compute predictions y_hat = w*x + b
        predictions = model(inputs)

        # 3) Compute loss (how far predictions are from targets)
        loss = loss_fn(predictions, targets)

        # 4) Numerical safety check: if loss becomes NaN/Inf (non-finite),
        #    abort the loop to avoid propagating invalid values.
        if not torch.isfinite(loss):
            break

        # 5) Backward pass: compute gradients via automatic differentiation
        loss.backward()

        # 6) Update parameters (w, b) using Adam
        optimizer.step()

    return model


def plot_results(
    x: list[float],
    test_input_features: torch.Tensor,
    test_output_features: torch.Tensor,
    test_predictions: torch.Tensor,
) -> None:
    target_input_features_list = test_input_features.detach().cpu().squeeze().tolist()
    target_output_features_list = test_output_features.detach().cpu().squeeze().tolist()
    test_predictions_list = test_predictions.detach().cpu().tolist()

    plt.figure(figsize=(7, 4))
    plt.title("Test domain: target vs prediction over x ∈ [0, 2π]")
    plt.xlabel("x (radians)")
    plt.ylabel("y")

    plt.plot(
        x, target_input_features_list, label="target inputs (cos(x))", color="tab:green"
    )
    plt.plot(
        x,
        target_output_features_list,
        label="target outputs (cos(x)+offset)",
        color="tab:blue",
    )
    plt.scatter(
        x, test_predictions_list, label="predictions", color="tab:orange", marker="x"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1) Build the dataset on the chosen device (MPS here for Apple Silicon)
    training_input_features, training_output_features = (
        make_training_input_and_output_features()
    )

    # 2) Train a simple model: a single linear neuron y = w*x + b
    model = train_linear_model(training_input_features, training_output_features)

    # 3) Test the model on a new dataset
    test_input_features, test_output_features, x = make_test_input_and_output_features()

    with torch.no_grad():
        test_predictions = model(test_input_features).squeeze()

    # 4) Graph the results
    plot_results(x, test_input_features, test_output_features, test_predictions)


if __name__ == "__main__":
    main()
