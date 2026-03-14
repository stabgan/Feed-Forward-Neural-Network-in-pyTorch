# Feed-Forward Neural Network in PyTorch

A simple 4-layer feed-forward neural network for MNIST digit classification, written in PyTorch.

## What This Does

Trains a fully connected neural network on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset (60k training / 10k test images). The model classifies 28×28 grayscale images into digits 0–9.

## Architecture

```
Input (784) → FC+ReLU (100) → FC+ReLU (100) → FC+ReLU (100) → FC (10)
```

| Component       | Detail              |
|-----------------|---------------------|
| Input           | 784 (28×28 flattened) |
| Hidden layers   | 3 × 100 units, ReLU |
| Output          | 10 classes (softmax via CrossEntropyLoss) |
| Optimizer       | SGD, lr = 0.1       |
| Batch size      | 100                 |
| Iterations      | 3000 (~5 epochs)    |

## Requirements

- Python 3.x
- PyTorch
- torchvision

```
pip install torch torchvision
```

## Usage

```bash
python fnn.py
```

MNIST data is downloaded automatically to `./data/` on first run. Training prints accuracy on the test set every 500 iterations.

## Known Issues and Deprecations

This code was written circa 2018 and targets an older version of PyTorch. Running it on modern PyTorch (≥ 0.5) will produce warnings or errors:

1. **`torch.autograd.Variable` is deprecated.** Since PyTorch 0.4, tensors track gradients natively. All `Variable(...)` wrapping is unnecessary.

2. **`loss.data[0]` crashes on modern PyTorch.** Scalar tensors no longer support indexing. Replace with `loss.item()`.

3. **Evaluation loop hardcodes `.cuda()` without a GPU check.** The training loop correctly gates on `torch.cuda.is_available()`, but the test/evaluation loop inside the training step does not — it calls `.cuda()` unconditionally. This will crash on CPU-only machines.

4. **`iter` shadows the Python built-in.** The variable name `iter` overwrites Python's built-in `iter()` function. Not a runtime error, but bad practice.

5. **Copy-paste error in comments.** The `forward()` method comments label the third linear layer and activation as "Linear function 2" / "Non-linearity 2" instead of 3.

## License

MIT — Kaustabh Ganguly, 2018
