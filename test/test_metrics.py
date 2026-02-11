import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from lps_ml.utils.metrics import Metric

# -----------------------------
# 1. Modelo fake (só para teste)
# -----------------------------
class DummyModel(nn.Module):
    def forward(self, x):
        return torch.rand(len(x))  # saída binária fake

model = DummyModel()

# -----------------------------
# 2. Dataset fake
# -----------------------------
x = torch.randn(20, 10)
y = torch.randint(0, 2, (20,))

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=5)

# -----------------------------
# 3. Avaliação manual (como você já fez)
# -----------------------------

#### Computar lista de metricas
Fazer chamada de funcoes estaticas que foram realocadas para metricas
targets = []
preds = []

with torch.no_grad():
    for xb, yb in dataloader:
        out = model(xb)
        preds.extend((out > 0.5).long().tolist())
        targets.extend(yb.tolist())

acc = Metric.ACCURACY.compute(targets, preds)
print(acc)
