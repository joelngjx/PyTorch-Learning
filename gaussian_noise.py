import matplotlib.pyplot as plt
from torch import nn
import torch
import random
import numpy as np


### 1. Preparing data (Using the linear function: Y = 2.5X + 1)
weight, bias = 2.5, 1
start, end, step = 0, 100, 0.5
X = torch.arange(start, end, step).unsqueeze(dim=1)
y_linear = (weight * X) + bias


## Manual implementation of Gaussian noise
noise_list = []
elements = torch.numel(X) # same num of elements

for sample in X:
    mean, std = 0, 0.5  # Mean is usually 0 for a std Gaussian distrib
    gauss_noise = random.gauss(mean, std)  # returns (random_num - mu) / sigma
    noise_list.append(gauss_noise)

noise_tensor = torch.tensor(noise_list).unsqueeze(dim=1)
y_gauss = (weight * X) + bias + noise_tensor # Labels


# Matplotlib visualisation
plt.figure()
plt.title("Linear Function with Gaussian Noise")
plt.scatter(X.numpy(), y_gauss.numpy(), alpha=0.2, label="Gaussian noise", c="r")
plt.plot(X.numpy(), y_linear.numpy(), c="b", label="Linear function")
plt.legend()
plt.show()


## Train-test split
split = int(0.8 * (len(X)))
X_train, y_train = X[:split], y_gauss[:split]
X_test, y_test = X[split:], y_gauss[split:]

# Visualisation function
def plot_predictions(train_d=X_train,
                     train_l=y_train,
                     test_d=X_test,
                     test_l=y_test,
                     predictions=None):
    plt.figure(figsize=(12,10))

    # Plot training and testing data
    plt.scatter(train_d, train_l, c="blue", label="Training data")
    plt.scatter(test_d, test_l, c="green", label="Testing data")

    plt.title("Predictions vs Training & Testing Data")

    if predictions is not None:
        plt.scatter(test_d, predictions, c="red", alpha=0.5, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


### 2. Building a Model
class LinearRegressionModel(nn.Module):
    def __init__ (self):
        super().__init__()

        # Setting up params
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
    
model = LinearRegressionModel()



### 3. Training the Model
# Loss function
# For a regression problem with Gaussian noise, we'll be using MSELoss 
loss_fn = nn.MSELoss()

# Optimiser
# We use a very small lr to avoid divergence due to squared errors on large inputs
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=1e-6)

## Setting up the training loop
torch.manual_seed(42)
epochs = 1001

# Lists to track progress
epoch_count = []
loss_values = []
test_loss_values = []

# Loop
for epoch in range(epochs):
    model.train()
    y_preds = model(X_train) # Forward pass
    loss = loss_fn(y_preds, y_train) # Calc. loss
    optimizer.zero_grad() # Zeroes optimiser gradients
    loss.backward()
    optimizer.step()
    model.eval()

    # Loss values for testing data
    with torch.inference_mode():
        test_preds = model(X_test)
        test_loss = loss_fn(test_preds, y_test)

    if epoch % 100 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

print(model.state_dict())
print(model.parameters())



### 4. Evaluation -> plotting test predictions
with torch.inference_mode():
    y_preds_new = model(X_test)


plot_predictions(predictions=y_preds_new.numpy())


## Plotting Loss Curves
plt.figure()
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Training loss")
plt.plot(epoch_count, test_loss_values, label="Testing Loss")
plt.legend()
plt.title("Loss Curves")
plt.show()
