import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import einops
import random

import multiprocessing
import wandb
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

from base import Unet, Transformer
from viz import tensor2gomuboard

class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.base_path = Path("./dataset/processed")
        self.x, self.y = self._load(max_idx)
        
        self.total = self.x.shape[0] // 12 - 1
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
        randomnum = random.randint(0, 12)
        x, y = self.x[idx*12+randomnum], self.y[idx*12+randomnum]
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        turns = torch.sum(x!=0)
        is_white = not bool(turns % 2)
        x *= 1 if is_white else -1

        x = einops.rearrange(x, "h w -> 1 h w")
        y = einops.rearrange(y, "h w -> 1 h w")
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return self.total

device = "cuda" if torch.cuda.is_available() else "cpu"
full_dataset = GOMUDataset(8000, device=device)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 50
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

nrow, ncol = 20, 20
channels = [1, 8, 20]
net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
# net = Transformer(1, 1, (20, 20), 16, 64).to(device)
learning_rate = 0.003
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.1)
total_parameters = sum([p.nelement() for p in net.parameters()])
print(total_parameters)

save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
save_base_path.mkdir(exist_ok=True)
(save_base_path / "ckpt").mkdir(exist_ok=True)
(save_base_path / "evalresults").mkdir(exist_ok=True)

cross_loss = nn.CrossEntropyLoss()
mse = nn.MSELoss()
alpha = 0.8
grad_clip=2

model_cfg = {"channels": channels, "nrow": nrow, "ncol": ncol}
exp_cfg = {"learning_rate": learning_rate, "train_size": train_size, "test_size": test_size, "alpha": alpha, "grad_clip": grad_clip}
    
run = wandb.init(
  project="AlphaGomu",
  config={**model_cfg, **exp_cfg, "param": total_parameters}
)

def get_loss(pred, GT):
    flatten_pred = einops.rearrange(pred, "b c h w -> b (c h w)")
    y = einops.rearrange(GT, "b c h w -> b (c h w)")
    y = y.argmax(-1)
    cross_en_loss = cross_loss(flatten_pred, y)
    mse_loss = mse(pred, GT)
    output = alpha*cross_en_loss + (1-alpha)*mse_loss
    return output

def one_loop(loader, net, optimizer, is_training, epoch):
    _loss = []
   
    if is_training:
        net.train()
        for (X, Y) in tqdm.tqdm(loader, desc="Training..."):
            pred = net(X)
            output = get_loss(pred, Y)
            output.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _loss.append(output.item())
    else:
        with torch.no_grad():
            net.eval()
            for (X, Y) in tqdm.tqdm(loader, desc="Testing..."):
                pred = net(X)
                output = get_loss(pred, Y)
                _loss.append(output.item())

            pred_pos_pil = tensor2gomuboard(pred[0], nrow, ncol, softmax=True, scale=10)
            # situation_pil = tensor2gomuboard(X[0], nrow, ncol)
            ground_true_pil = tensor2gomuboard(Y[0]*2+X[0], nrow, ncol)
            eval_result_path = save_base_path / f"evalresults/epoch-{epoch}.png"
            eval_gt_path = save_base_path / f"evalresults/gt-{epoch}.png"
            pred_pos_pil.save(eval_result_path)
            ground_true_pil.save(eval_gt_path)

    return sum(_loss) / len(_loss)

train_losses = []
test_losses = []
nb_epoch = 100
for epoch in range(nb_epoch):
    train_loss = one_loop(train_loader, net, optimizer, True, epoch)
    test_loss = one_loop(test_loader, net, optimizer, False, epoch)

    train_losses.append(train_loss)
    test_losses.append(test_losses)

    run.log({"train/loss": train_loss})
    run.log({"test/loss": test_loss})

    save_path = save_base_path / f"ckpt/epoch-{epoch}.pkl"
    torch.save({"model": net.state_dict(), "optim": optimizer.state_dict()}, save_path)

    print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")


# fig, ax = plt.subplots(1, 2)

x = np.arange(nb_epoch)
plt.plot(x, train_losses)
plt.plot(x, test_losses)
plt.savefig("base_training_results.png", dpi=500)
plt.show()
# ax[0].plot(x, train_losses)
# ax[1].plot(x, test_losses)
# fig.savefig("base_training_results.png", dpi=500)