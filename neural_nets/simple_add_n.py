#!/usr/bin/env python3
"""Minimal PyTorch example: learn an affine mapping y = x + c with a tiny network.

This script builds a single linear neuron (`nn.Linear(1, 1)`) and trains it with
mean squared error to recover a constant-offset mapping of the form y = x + c.
The toy dataset is produced by `make_inputs_and_targets()`; with the current
settings it creates 100 points and an offset c = target_start - input_start.
The program runs a short training loop (Adam) and prints learned parameters,
loss, predictions, and a simple visualization.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("mps")


def make_inputs_and_targets():
    input_start = 10
    target_start = input_start + 10
    count = 100
    step = 1

    inputs = (
        torch.arange(input_start, input_start + count, step=step, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )
    targets = (
        torch.arange(target_start, target_start + count, step=step, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    return inputs, targets


def main() -> None:
    # 1) Build the dataset on the chosen device (MPS here for Apple Silicon)

    inputs, targets = make_inputs_and_targets()
    offset = (targets[0] - inputs[0]).item()

    # 2) Define a simple model: a single linear neuron y = w*x + b
    nn_linear = nn.Linear(1, 1)
    linear_model = nn.Sequential(nn_linear).to(device)

    # Initialize near the expected solution to make convergence obvious
    with torch.no_grad():
        linear_model[0].weight.fill_(1.0)
        linear_model[0].bias.zero_()

    # 3) Choose a loss function and optimizer
    loss_function = nn.MSELoss()  # mean squared error loss function
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.1)  # Adam optimizer

    # 4) Training loop: forward -> loss -> backward -> step
    for _ in range(500):
        optimizer.zero_grad()  # reset gradients
        predictions = linear_model(inputs)  # forward pass
        loss = loss_function(predictions, targets)  # compute loss
        if not torch.isfinite(loss):
            break
        loss.backward()  # backward pass
        optimizer.step()  # update parameters

    # 5) Inspect learned parameters and predictions

    with torch.no_grad():
        new_inputs, _ = make_inputs_and_targets()
        predictions = linear_model(new_inputs).squeeze()

    predictions_list = predictions.tolist()

    learned_weight = linear_model[0].weight.item()
    learned_bias = linear_model[0].bias.item()

    print(
        {
            "learned_weight": learned_weight,
            "learned_bias": learned_bias,
            "expected_bias": offset,
            "loss": round(float(loss.item()), 6),
        }
    )

    print("preds:", predictions_list)

    # 7) Simple visualization (requires matplotlib)
    # Move tensors to CPU and convert to lists for plotting

    plt.figure(figsize=(6, 4))
    plt.title(f"Learning y = x + {offset:.0f} (Linear Model)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Prepare simple Python lists for plotting
    inputs_list = inputs.squeeze().tolist()
    targets_list = targets.squeeze().tolist()

    # Ground truth points (targets)
    plt.scatter(inputs_list, targets_list, label="target", color="tab:blue")
    # Model predictions
    plt.plot(
        inputs_list,
        predictions_list,
        label="prediction",
        color="tab:orange",
        marker="o",
    )
    # Also show the raw inputs (identity mapping y = x) for context
    plt.plot(
        inputs_list, inputs_list, label="input (y=x)", color="tab:green", linestyle="--"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
