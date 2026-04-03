import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import PathTracerDataset, PathTracerModel

epochs = 8
batch_size = 1
width, height = 1280, 720

train_set = PathTracerDataset(train=True)
test_set = PathTracerDataset(train=False)
train_loader = DataLoader(
    train_set, batch_size=batch_size, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, num_workers=4, pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = PathTracerModel()
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))

model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

for epoch in range(epochs):
    model.train()
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(X.to(device, non_blocking=True))

        loss = loss_fn(out, y.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()

        print("                                                         ", end="\r")
        print(f"Step {i + 1}/{len(train_loader)}: Loss={loss.item():.6f}", end="\r")

    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X.to(device))
            loss = loss_fn(out, y.to(device))
            test_loss += loss.item()

    print("                                                             ", end="\r")
    print(f"Epoch {epoch + 1}/{epochs}:\n  Loss={test_loss / len(test_loader):.6f}")

    img = out.cpu()[0].permute(1, 2, 0).detach().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save("latest_out.png")
    img = y[0].permute(1, 2, 0).detach().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save("latest_y.png")

    torch.save(model.state_dict(), "model.pth")
