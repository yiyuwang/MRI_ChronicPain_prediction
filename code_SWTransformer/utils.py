import torch
import torch.nn.functional as F
import numpy as np

def get_model_dir(run_name, root_path):
    import os
    os.makedirs(os.path.join(root_path, 'models_save', run_name), exist_ok=True)
    return os.path.join(root_path, 'models_save', run_name)

def extract_epoch(path):
    import re
    match = re.search(r'epoch-(\d+).pt', path)
    if match:
        return int(match.group(1))
    else:
        return -1


def get_optimizer(model, lr, weight_decay):
        """Get optimizers."""
        # we don't apply weight decay to bias and layernorm parameters, as inspired by:
        # https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb
        no_decay = ["bias", "LayerNorm.weight"] # in case we upgrade to Transformer arch
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ] 
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        return optimizer

def compute_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)  # Find the predicted class index
    correct = (predicted == y_true).float().sum().item()
    return correct 

def compute_auc(y_preds, y_trues):
    from sklearn.metrics import roc_auc_score
    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    predicted_prob, _ = torch.max(y_preds, 1)
    auc = roc_auc_score(y_trues.detach().cpu().numpy(), predicted_prob.detach().cpu().numpy())
    return auc

## Lambda function for learning rate adjustment
def get_lr_lambda(total_epochs=200, warmup_epochs=5, factor=10, lr=1):

    # here lr is just a multiplicative factor 1
    start_lr = lr / factor
    peak_lr = lr
    end_lr = lr / factor

    # Lambda function for learning rate adjustment
    def lr_lambda(epoch):
        # Linear warm-up
        if epoch < warmup_epochs:
            return (peak_lr - start_lr) / warmup_epochs * epoch + start_lr
        # Linear decay
        decay_epochs = total_epochs - warmup_epochs
        return peak_lr - (peak_lr - end_lr) * (epoch - warmup_epochs) / decay_epochs
    return lr_lambda



def smoothed_labels_to_age(smoothed_labels, min_age=20):
    """ convert predicted age (as smoothed out tensor) to single age"""
    # Option 1: Take argmax as predicted age
    # ages = smoothed_labels.argmax(dim=-1) + min_age

    # Option 2: Compute expected value
    probs = F.softmax(smoothed_labels, dim=-1)
    expected_value = (probs * torch.arange(min_age, min_age + probs.size(-1)).float()).sum(dim=-1)
    ages = expected_value
    return ages


from sklearn import metrics


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


