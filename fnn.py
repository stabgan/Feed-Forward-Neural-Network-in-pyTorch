"""Feed-Forward Neural Network for MNIST Classification.

A 4-layer fully connected network trained on the MNIST dataset
to classify handwritten digits (0-9).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class FeedforwardNeuralNetModel(nn.Module):
    """4-layer feed-forward network: 784 → 100 → 100 → 100 → 10."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        return out


def train(model, device, train_loader, test_loader, criterion, optimizer, n_iters):
    """Train the model and evaluate every 500 iterations."""
    iter_count = 0
    num_epochs = int(n_iters / (len(train_loader.dataset) / train_loader.batch_size))

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter_count += 1

            if iter_count % 500 == 0:
                accuracy = evaluate(model, device, test_loader)
                print(
                    f"Iteration: {iter_count}. "
                    f"Loss: {loss.item():.4f}. "
                    f"Accuracy: {accuracy:.2f}%"
                )
                model.train()


def evaluate(model, device, test_loader):
    """Evaluate model accuracy on the test set."""
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main():
    # ── Hyperparameters ──────────────────────────────────────────
    batch_size = 100
    n_iters = 3000
    learning_rate = 0.1
    input_dim = 28 * 28
    hidden_dim = 100
    output_dim = 10

    # ── Dataset ──────────────────────────────────────────────────
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # ── Model / Loss / Optimizer ─────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # ── Train ────────────────────────────────────────────────────
    train(model, device, train_loader, test_loader, criterion, optimizer, n_iters)

    # ── Final evaluation ─────────────────────────────────────────
    final_accuracy = evaluate(model, device, test_loader)
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
