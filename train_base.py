import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import einops
import multiprocessing

from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

from base import Unet

class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.base_path = Path("./dataset/processed")
        self.x, self.y = self._load(max_idx)
        
        self.total = self.x.shape[0]
        # self.total = max_idx

    def _load(self, max_idx):
        inputs = []
        outputs = []

        for i in tqdm.tqdm(range(max_idx+1)):
            # inputs, outputs = p.map(self._load_file, list(range(max_idx+1)))
            _inputs, _outputs = self._load_file(i)
            inputs.append(_inputs)
            outputs.append(_outputs)
                
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)
        
        return inputs, outputs
   
    def _load_file(self, i):
        file_path = self.base_path / f"{i:05d}.npz"
        data = np.load(file_path)
        
        _inputs = data["inputs"]
        _outputs = data["outputs"]
        
        return _inputs, _outputs

    def __getitem__(self, idx):
        # x, y = self._load_file(idx)
        x, y = self.x[idx], self.y[idx]
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        turns = torch.sum(x!=0)
        is_white = not bool(turns % 2)
        x *= 1 if is_white else -1

        x = einops.rearrange(x, "h w -> 1 h w")
        y = einops.rearrange(y, "h w -> 1 h w")
        y = y.argmax()
        x = x.to(torch.float32)

        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return self.total

device = "cuda" if torch.cuda.is_available() else "cpu"
full_dataset = GOMUDataset(8000, device=device)
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

save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")

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

    save_path = save_base_path / f"{epoch}.pkl"
    torch.save({"model": net.state_dict(), "optim": optimizer.state_dict()}, )

    print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")


fig, ax = plt.subplots(1, 2)

x = np.arange(nb_epoch)
ax[0].plot(x, train_losses)
ax[1].plot(x, test_losses)
fig.savefig("base_training_results.png", dpi=500)