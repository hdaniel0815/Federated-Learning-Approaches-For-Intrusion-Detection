import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



########################################
# utils
########################################


@torch.no_grad()
def infer(model, loader, device):
    model = model.to(device)
    model.eval()
    ys, preds = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        ys.append(y.cpu())
        preds.append(logits.argmax(dim=1).cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Save a labelled confusion matrix heatmap to save_path."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    fig_size = max(8, len(class_names))
    plt.figure(figsize=(fig_size, max(6, fig_size - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()