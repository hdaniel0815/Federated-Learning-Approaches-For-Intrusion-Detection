import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



########################################
# utils
########################################


@torch.no_grad()
def infer(model, loader, device):
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
        "acc": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }