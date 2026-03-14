# Feed-Forward Neural Network in PyTorch

A minimal 4-layer fully connected network that classifies handwritten digits from the MNIST dataset, achieving ~97 % test accuracy.

## What It Does

Trains a feed-forward neural network on [MNIST](http://yann.lecun.com/exdb/mnist/) (60 k training / 10 k test grayscale 28×28 images) to recognize digits 0–9. Test accuracy is printed every 500 iterations.

## Architecture

```
Input (784) → FC + ReLU (100) → FC + ReLU (100) → FC + ReLU (100) → FC (10)
```

| Component       | Detail                              |
|-----------------|-------------------------------------|
| Input           | 784 (28×28 flattened)               |
| Hidden layers   | 3 × 100 units, ReLU activation      |
| Output          | 10 classes, CrossEntropyLoss        |
| Optimizer       | SGD, lr = 0.1                       |
| Batch size      | 100                                 |
| Training        | 3 000 iterations (~5 epochs)        |

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 🔥 PyTorch ≥ 2.0 | Deep learning framework |
| 🖼️ torchvision ≥ 0.15 | MNIST dataset and transforms |
| 🧮 CUDA (optional) | GPU acceleration |

## Getting Started

```bash
pip install -r requirements.txt
python fnn.py
```

MNIST downloads automatically to `./data/` on first run.

## Project Structure

```
├── fnn.py              # Model definition, training, and evaluation
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT
└── README.md
```

## ⚠️ Known Issues

- No learning-rate scheduler — accuracy may plateau with longer training.
- No model checkpointing or saving.
- Single fixed architecture; no hyperparameter search.

## License

MIT
