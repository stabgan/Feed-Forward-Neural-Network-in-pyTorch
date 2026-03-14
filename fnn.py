# Feed-Forward Neural Network for MNIST Classification
# Dependencies
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

'''
STEP 1: LOADING DATASET
'''

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
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
        # Linear function 1
        out = self.fc1(x)
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        out = self.relu2(out)

        # Linear function 3
        out = self.fc3(out)
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''

criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''

iter_count = 0
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter_count += 1

        if iter_count % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0

            model.eval()
            with torch.no_grad():
                for test_images, test_labels in test_loader:
                    test_images = test_images.view(-1, 28 * 28).to(device)
                    test_labels = test_labels.to(device)

                    # Forward pass only to get logits/output
                    test_outputs = model(test_images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(test_outputs.data, 1)

                    # Total number of labels
                    total += test_labels.size(0)

                    # Total correct predictions
                    correct += (predicted == test_labels).sum().item()

            accuracy = 100.0 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {:.4f}. Accuracy: {:.2f}%'.format(
                iter_count, loss.item(), accuracy))

            model.train()
