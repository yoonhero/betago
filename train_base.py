import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import einops

from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from base import Unet

class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.x, self.y = self._load(max_idx)
        self.total = self.x.shape[0]
    
    def _load(self, max_idx):
        inputs = np.array([])
        outputs = np.array([])

        base_path = Path("./dataset/processed")
        for i in range(max_idx+1):
            file_path = base_path / f"{i:05d}.npz"
            data = np.load(file_path)

            _inputs = data["inputs"]
            _outputs = data["outputs"]

            if inputs.shape == (0,):
                inputs = _inputs.copy()
                outputs = _outputs.copy()
                continue

            inputs = np.concatenate([inputs, _inputs])
            outputs = np.concatenate([outputs, _outputs])
            
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        return inputs, outputs

    def __getitem__(self, idx):
        turns = torch.sum(self.x[idx]!=0)
        is_white = not bool(turns % 2)
        x = self.x[idx] if is_white else -self.x[idx]
        y = self.y[idx]

        x = einops.rearrange(x, "h w -> 1 h w")
        y = einops.rearrange(y, "h w -> 1 h w")
        y = y.argmax()

        return x, y

    def __len__(self):
        return self.total

device = "cuda" if torch.cuda.is_available() else "cpu"
full_dataset = GOMUDataset(14087, device=device)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

nrow, ncol = 20, 20
channels = [1, 4, 16]
net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def one_loop(loader, net, optimizer, is_training=True):
    _loss = []
    loss = nn.CrossEntropyLoss()
    
    if is_training:
        for (X, Y) in tqdm.tqdm(loader, desc="Training..."):
            pred = net(X)
            pred = einops.rearrange(pred, "b c h w -> b (c h w)")
            output = loss(pred, Y)
            output.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _loss.append(output.item())
    else:
        with torch.no_grad():
            for (X, Y) in tqdm.tqdm(loader, desc="Testing..."):
                pred = net(X)
                pred = einops.rearrange(pred, "b c h w -> b (c h w)")
                output = loss(pred, Y)
                optimizer.step()

    return sum(_loss) / len(_loss)

train_losses = []
test_losses = []

nb_epoch = 40
for epoch in range(nb_epoch):
    train_loss = one_loop(train_loader, net, optimizer, True)
    test_loss = one_loop(test_loader, net, optimizer, False)

    train_losses.append(train_loss)
    test_losses.append(test_losses)

    print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")


fig, ax = plt.subplots(1, 2)

x = np.arange(nb_epoch)
ax[0].plot(x, train_losses)
ax[1].plot(x, test_losses)
fig.savefig("base_training_results.png", dpi=500)