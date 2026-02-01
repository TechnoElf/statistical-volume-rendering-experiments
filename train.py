import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import PathTracerDataset, PathTracerModel

epochs = 1
batch_size = 1
width, height = 1280, 720

train_set = PathTracerDataset(train=True)
test_set = PathTracerDataset(train=False)
train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

model = PathTracerModel()
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(epochs):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(X)

        img = out[0].permute(1, 2, 0).detach().numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save("latest_out.png")
        img = y[0].permute(1, 2, 0).detach().numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save("latest_y.png")

        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        print("                                                         ", end="\r")
        print(f"Step {i + 1}/{len(train_loader)}: Loss={loss.item():.4f}", end="\r")

        torch.save(model.state_dict(), "model.pth")

    test_loss = 0

    model.eval()
    for X, y in test_loader:
        out = model(X)
        loss = loss_fn(out, y)
        test_loss += loss.item()

    print("                                                             ", end="\r")
    print(f"Epoch {epoch + 1}/{epochs}:\n  Loss={test_loss / len(test_loader):.4f}")
