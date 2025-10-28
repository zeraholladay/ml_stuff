#!/usr/bin/env python3
import numpy as np

def make_batch(rng, n_samples, seq_min=3, seq_max=10, n=7, low=-10, high=10):
    lengths = rng.integers(seq_min, seq_max+1, size=n_samples)
    max_len = lengths.max()
    X = np.zeros((n_samples, max_len), dtype=np.float32)
    Y = np.zeros_like(X)
    M = np.zeros_like(X, dtype=np.float32)
    for i, L in enumerate(lengths):
        seq = rng.integers(low, high+1, size=L).astype(np.float32)
        X[i, :L] = seq
        Y[i, :L] = seq + n
        M[i, :L] = 1.0
    return X, Y, M

def train_add_n(n=7, seed=123, n_train=2000, n_val=400, epochs=3000, lr=5e-3):
    rng = np.random.default_rng(seed)
    X_train, Y_train, M_train = make_batch(rng, n_train, n=n)
    X_val, Y_val, M_val = make_batch(rng, n_val, n=n)

    mean_x = np.sum(X_train * M_train) / np.sum(M_train)
    std_x = np.sqrt(np.sum(((X_train - mean_x)*M_train)**2) / np.sum(M_train))
    X_train_n = (X_train - mean_x) / (std_x + 1e-8)
    X_val_n = (X_val - mean_x) / (std_x + 1e-8)

    w = np.array(1.0 + rng.normal(0, 0.01), dtype=np.float32)
    b = np.array(0.0, dtype=np.float32)

    def forward(xn):
        return w * ((xn * std_x) + mean_x) + b

    def step(Xn, Y, M):
        pred = forward(Xn)
        denom = np.sum(M)
        dL_dpred = 2.0 * (pred - Y) * M / denom
        grad_w = np.sum(dL_dpred * ((Xn * std_x) + mean_x))
        grad_b = np.sum(dL_dpred)
        return grad_w, grad_b

    for _ in range(epochs):
        gw, gb = step(X_train_n, Y_train, M_train)
        w -= lr * gw
        b -= lr * gb

    # Eval
    def mse(pred, target, mask):
        diff = (pred - target) * mask
        return float(np.sum(diff*diff) / np.sum(mask))

    train_mse = mse(forward(X_train_n), Y_train, M_train)
    val_mse = mse(forward(X_val_n), Y_val, M_val)
    return {"w": float(w), "b": float(b), "train_mse": train_mse, "val_mse": val_mse,
            "mean_x": float(mean_x), "std_x": float(std_x)}

if __name__ == "__main__":
    res = train_add_n(n=7)
    print(res)
