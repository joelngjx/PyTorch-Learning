# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


### 1. PREPROCESSING DATA
# Convert data to Tensors 
'''
Note: MNIST images are PIL images, we have to convert them to PT tensors
-> transforms.Compose() takes a list of transformations

* Obtain values for scaling (to [0-1]) and normalizing (gaussian distrib w/ mean=0, std=1)
train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(train.data.min(), train.data.max()))
print('Mean Pixel Value {} \nPixel Values Std: {}'.format(train.data.float().mean(), train.data.float().std()))

* Scaled mean (0.1306...) & scaled std (0.3081...)
print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(train.data.float().mean() / 255, train.data.float().std() / 255))
'''

transformData = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1306,), std=(0.3081,))
])


# Set up training & testing set
train = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,  # Rmb to set False after downloading
    transform=transformData
)

test = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transformData
)


## Get FEATURES and LABELS
X_train, y_train, X_test, y_test = [], [], [], []
for img, label in train:
    X_train.append(img)
    y_train.append(label)

for img, label in test:
    X_test.append(img)
    y_test.append(label)

# Convert features and labels to tensors
X_train, X_test = torch.stack(X_train), torch.stack(X_test)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)



### 2. BUILDING A MODEL
# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Model
from torch import nn
import torchmetrics
class MNISTModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 10):
        super().__init__()

        self.linearLayerStack = nn.Sequential(
            nn.Flatten(),  # turns [1, 28, 28] -> [784] 
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linearLayerStack(x)


# Instantiating
model = MNISTModel(
    input_features=784,
    output_features=10
).to(device)


# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Accuracy function
def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct/(len(y_pred))*100)


from torchmetrics.functional import precision, recall, f1_score
# Precision function 
# P = TP / (TP + FP)
def precision_fn(y_logits, y_true):
    return(precision(y_logits, y_true, task="multiclass", num_classes=10)*100)


# Recall function
# R = TP / (TP + FN)
def recall_fn(y_logits, y_true):
    return(recall(y_logits, y_true, task="multiclass", num_classes=10)*100)


# F1Score function
# F1 = (2*P*R)/(P+R)
def f1_fn(y_logits, y_true):
    return(f1_score(y_logits, y_true, task="multiclass", num_classes=10)*100)




### 3. TRAINING LOOP
torch.manual_seed(42)
epochs = 21

## Creating DataLoaders and splitting into batches
from torch.utils.data import DataLoader, TensorDataset
train_dataset, test_dataset = TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
train_loader, test_loader = DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64)


# Loop
for epoch in range (epochs):
    ## Training
    model.train()
    train_loss, train_acc = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_logits = model(X_batch)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # Loss and Acc
        loss = loss_fn(y_logits, y_batch)   # conceptually similar as in test set but required for backpropagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y_preds, y_batch)


    ## Testing
    model.eval()
    test_loss, test_acc = 0, 0
    all_test_logits, all_test_labels = [], []
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            test_logits = model(X_batch)
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            # Test Loss and Acc
            test_loss += loss_fn(test_logits, y_batch).item()
            test_acc += accuracy_fn(test_preds, y_batch)

            ## APPEND values for final metrics
            all_test_logits.append(test_logits.cpu())
            all_test_labels.append(y_batch.cpu())


    if epoch % 10 == 0:
        print(f"Epoch {epoch} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc/len(train_loader):.2f}% | "
              f"Test Loss: {test_loss/len(test_loader):.4f} | "
              f"Test Acc: {test_acc/len(test_loader):.2f}%")


# Final metrics
all_test_logits = torch.cat(all_test_logits)
all_test_labels = torch.cat(all_test_labels)

print(f"Test Precision: {precision_fn(all_test_logits, all_test_labels):.2f}%")
print(f"Test Recall: {recall_fn(all_test_logits, all_test_labels):.2f}%")
print(f"Test F1: {f1_fn(all_test_logits, all_test_labels):.2f}%")



#### VISUALISATION
from torchmetrics.classification import MulticlassPrecisionRecallCurve

# Put model in eval mode and compute logits
model.eval()
with torch.inference_mode():
    test_logits = model(X_test)


### PR CURVES (PER CLASS) ###
# Initialize PR curve calculator
pr_curve = MulticlassPrecisionRecallCurve(num_classes=10)
# Get precision, recall, thresholds for each class
precisions, recalls, thresholds = pr_curve(test_logits, y_test)


### MICRO-AVERAGE CURVE ###
# NOTE that thresholds are not required (we compute PR directly without storing thresholds for various classes)
micro_curve = MulticlassPrecisionRecallCurve(num_classes=10, average="micro")
precisions_micro, recalls_micro, na = micro_curve(test_logits, y_test)


plt.figure(figsize=(14,6))


## Subplot 1
plt.subplot(1, 2, 1)
for i in range(10):
    plt.plot(recalls[i].cpu(), precisions[i].cpu(), label=f"Class {i}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (per class)")
plt.legend()
plt.grid(True)

# Subplot 2
plt.subplot(1, 2, 2)
plt.plot(recalls_micro.cpu(), precisions_micro.cpu(), color="black", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Micro-average Precision–Recall Curve")
plt.grid(True)


plt.tight_layout()
plt.show()