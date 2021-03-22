import torch
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score
from ..data.data_classes import EMPTY_FIELD_VALUE
import torch.nn as nn
import torch

## Losses


def L2_loss(out, target, w):
    # w : whether each sample has to be included
    # for instance, OU don't have to be included in alpha,
    # while LW don't have to be included in theta
    mean_dims = torch.mean(torch.pow(out[w] - target[w], 2), dim=1)
    # concerned = torch.masked_select(mean_dims,w)
    return torch.mean(mean_dims)


def Category_loss(out, target, w):
    # w : whether each sample has to be included
    # return nn.CrossEntropyLoss()(torch.index_select(out,0,w), torch.index_select(target,0,w))
    return nn.CrossEntropyLoss()(
        out[w],
        target[w].view(
            -1,
        ),
    )


## Metrics for training


def MAE_metric(out, info, field="alpha"):
    return np.mean(np.abs(out[field] - np.reshape(info[field], out[field].shape)))


def F1_metric(out, info, field="model"):
    return f1_score(
        y_true=info[field], y_pred=np.argmax(out[field], axis=1), average="micro"
    )


def is_concerned(target):
    return torch.eq(torch.sum(1 * torch.eq(target, EMPTY_FIELD_VALUE), dim=1), 0)
