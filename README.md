# Feed-Forward Neural Network in PyTorch

A minimal feed-forward neural network that classifies handwritten digits from the MNIST dataset.

## What It Does

Trains a 4-layer fully connected network on [MNIST](http://yann.lecun.com/exdb/mnist/) (60k train / 10k test grayscale images, 28×28 pixels) to recognize digits 0–9. Achieves ~97% test accuracy in ~3000 iterations.

## Architecture

```
Input (784) → FC+ReLU (100) → FC+ReLU (100) → FC+ReLU (100) → FC (10)
```

| Layer           | Detail                              |
|-----------------|-------------------------------------|
| Input           | 784 (28×28 flattened)               |
| Hidden layers   | 3 × 100 units, ReLU activation      |
| Output          | 10 classes (CrossEntropyLoss)       |
| Optimizer       | SGD, lr = 0.1                       |
| Batch size      | 100                                 |
| Training        | 3000 iterations (~5 epochs)         |

## Dependencies

- Python 3.7+
- PyTorch ≥ 1.0
- torchvision

```bash
pip install torch torchvision
```

## How to Run

```bash
python fnn.py
```

MNIST data downloads automatically to `./data/` on first run. Test accuracy is printed every 500 iterations.

## Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ torchvision | MNIST dataset + transforms |
| 🧮 CUDA | Optional GPU acceleration |

## Known Issues

- No learning rate scheduler — accuracy may plateau with longer training.
- Single fixed architecture; no hyperparameter search.
- No model checkpointing or saving.

## License

MIT
