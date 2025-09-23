from torchmetrics.functional import precision, recall, f1_score
import torch


# Accuracy function
def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    return (correct/(len(y_pred))*100)


# Precision function 
# P = TP / (TP + FP)
def precision_fn(y_logits, y_true):
    return(precision(y_logits, y_true, task="multiclass", num_classes=10, average="macro")*100)


# Recall function
# R = TP / (TP + FN)
def recall_fn(y_logits, y_true):
    return(recall(y_logits, y_true, task="multiclass", num_classes=10, average="macro")*100)


# F1Score function
# F1 = (2*P*R)/(P+R)
def f1_fn(y_logits, y_true):
    return(f1_score(y_logits, y_true, task="multiclass", num_classes=10, average="macro")*100)
