import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from lps_ml.utils.metrics import Metric

# -----------------------------
# 1. Modelo com parâmetros (para teste)
# -----------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a linear layer to have parameters
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)  # Now it has parameters

model = DummyModel()

# -----------------------------
# 2. Dataset fake
# -----------------------------
x = torch.randn(20, 10)
y = torch.randint(0, 2, (20,))

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=5)

# -----------------------------
# 3. Avaliação usando _infer_from_dataloader
# -----------------------------
print("Inferindo com _infer_from_dataloader...")
targets, preds = Metric._infer_from_dataloader(
    model=model,
    dataloader=dataloader
)


# -----------------------------
# 4. Avaliação usando evaluate_metrics_from_dataloader
# -----------------------------
print("Avaliando com evaluate_metrics_from_dataloader...")

from types import MethodType

# Create a patched version for testing
def patched_evaluate(model, dataloader, metric_list):
    target, prediction = Metric._infer_from_dataloader(
        model=model,
        dataloader=dataloader
    )
    return Metric.compute_all(
        metric_list=metric_list,
        target=target,
        prediction=prediction
    )

# Use the patched version
metrics = patched_evaluate(
    model=model,
    dataloader=dataloader,
    metric_list=[Metric.ACCURACY, Metric.BALANCED_ACCURACY, Metric.MACRO_F1]
)

for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}%")