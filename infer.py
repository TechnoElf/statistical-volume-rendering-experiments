import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import PathTracerDataset, PathTracerModel

batch_size = 1
width, height = 1280, 720

test_set = PathTracerDataset(train=False, directory="dataset")
test_loader = DataLoader(test_set, batch_size=batch_size)

model = PathTracerModel()
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))

loss_fn = nn.MSELoss()

model.eval()
X, y = test_loader.__iter__().__next__()

out = model(X)

img = out[0].permute(1, 2, 0).detach().numpy()
Image.fromarray((img * 255).astype(np.uint8)).save("latest_out.png")
img = y[0].permute(1, 2, 0).detach().numpy()
Image.fromarray((img * 255).astype(np.uint8)).save("latest_y.png")

loss = loss_fn(out, y)
