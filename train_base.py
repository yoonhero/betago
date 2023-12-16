#! /usr/bin/env python
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

from base import PocliyValueNet, Unet, Transformer, get_total_parameters
from viz import tensor2gomuboard

class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.base_path = Path("./dataset/processed")
        self.board_state, self.next_pos, self.winner = self._load(max_idx)
        
        self.total = self.board_state.shape[0] // 12 - 1
        # self.total = max_idx

    def _load(self, max_idx):
        inputs = []
        outputs = []
        game_results = []

        for i in tqdm.tqdm(range(max_idx+1)):
            # inputs, outputs = p.map(self._load_file, list(range(max_idx+1)))
            _inputs, _outputs, _results = self._load_file(i)
            inputs.append(_inputs)
            outputs.append(_outputs)
            game_results.append(_results)
                
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)
        game_results = np.concatenate(game_results)
        
        return inputs, outputs, game_results
   
    def _load_file(self, i):
        file_path = self.base_path / f"{i:05d}.npz"
        data = np.load(file_path, allow_pickle=True)
        
        _inputs = data["inputs"]
        _outputs = data["outputs"]
        _results = data["results"]

        return _inputs, _outputs, _results

    # Return [Board, NextPos, Win%]
    def __getitem__(self, idx):
        # x, y = self._load_file(idx)
        randomnum = random.randint(0, 12)
        to_get_idx = idx*12+randomnum
        board, next_pos, game_result = self.board_state[to_get_idx], self.next_pos[to_get_idx], self.winner[to_get_idx]
        # BLACK: 1, WHITE: -1: DRAW=0
        x, y = torch.from_numpy(board), torch.from_numpy(next_pos)

        # if your turn?
        # me in first row
        # comptetitor in second row
        total_BLACK = torch.sum(x[0])
        total_WHITE = torch.sum(x[1])
        if total_BLACK != total_WHITE:
            # print(total_BLACK, total_WHITE)
            x[0], x[1] = x[1], x[0]
            game_result = -game_result

        # WIN: 1 | DRAW: 0.5 | LOSE: 0
        game_result = (game_result+1)/2
        game_result = torch.from_numpy(game_result)

        # IMG [B, C, H, W]
       # x = einops.rearrange(x, "2 h w -> h w")
        y = einops.rearrange(y, "h w -> 1 h w")
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        game_result = game_result.to(torch.float32)

        return x.to(self.device), y.to(self.device), game_result.to(self.device)

    def __len__(self):
        return self.total

total_samples = 14000
device = "cuda" if torch.cuda.is_available() else "cpu"
full_dataset = GOMUDataset(total_samples, device=device)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

nrow, ncol = 20, 20
# channels = [2, 8, 36]
channels = [2, 64, 128, 256, 128, 64, 1]
#net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
net = PocliyValueNet(nrow, ncol, channels).to(device)
# net = Transformer(1, 1, (20, 20), 16, 64).to(device)
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
total_parameters = get_total_parameters(net)
print(total_parameters)

save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
save_base_path.mkdir(exist_ok=True)
(save_base_path / "ckpt").mkdir(exist_ok=True)
(save_base_path / "evalresults").mkdir(exist_ok=True)

cross_loss = nn.CrossEntropyLoss()
mse = nn.MSELoss()
bce = nn.BCELoss()
alpha = 0.8
grad_clip=2

model_cfg = {"channels": channels, "nrow": nrow, "ncol": ncol, "param": total_parameters}
exp_cfg = {"learning_rate": learning_rate, "train_size": train_size, "test_size": test_size, "alpha": alpha, "grad_clip": grad_clip, "total_samples": total_samples}
    
run = wandb.init(
  project="AlphaGomu",
  config={**model_cfg, **exp_cfg}
)

def get_loss(policy, value, GT, win):
    ypred = policy.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    # flatten_pred = einops.rearrange(policy, "b c h w -> b (c h w)")
    # y = einops.rearrange(GT, "b c h w -> b (c h w)")
    y = GT.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    y = y.argmax(-1)
    cross_en_loss = cross_loss(ypred, y)
    mse_loss = mse(policy, GT)
    policy_loss = alpha*cross_en_loss + (1-alpha)*mse_loss
    value_loss = bce(value, win)

    return policy_loss+value_loss

def one_loop(loader, net, optimizer, training, epoch):
    _loss = []
    num_correct = 0
    num_samples = 0

    if training:
        with tqdm.tqdm(loader) as pbar:
            for step, (X, Y, win) in enumerate(pbar):
                net.train()
                policy, value = net(X)
                loss = get_loss(policy, value, Y, win)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _loss.append(loss.item()) 

                pbar.set_description(f"{epoch}/{step}")
                pbar.set_postfix(loss=_loss[-1])
                
                num_correct += ((value>0.5)==win).sum()
                num_samples += value.size(0)

    if not training:
        with torch.no_grad():
            net.eval()
            for (X, Y, win) in tqdm.tqdm(loader, desc="Testing..."):
                policy, value = net(X)
                output = get_loss(policy, value, Y, win)
                _loss.append(output.item())

                num_correct += ((value>0.5)==win).sum()
                num_samples += value.size(0)

            pred_pos_pil = tensor2gomuboard(policy[0], nrow, ncol, softmax=True, scale=10)
            concatenated = X[0][0]-X[0][1]
            ground_true_pil = tensor2gomuboard(2*Y[0]+concatenated, nrow, ncol)
            eval_result_path = save_base_path / f"evalresults/{epoch}-pred.png"
            eval_gt_path = save_base_path / f"evalresults/{epoch}-gt.png"
            pred_pos_pil.save(eval_result_path)
            ground_true_pil.save(eval_gt_path)
            
    return sum(_loss) / len(_loss), num_correct / num_samples

train_losses = []
test_losses = []
nb_epoch = 20
for epoch in range(nb_epoch):
    train_loss, train_accuracy= one_loop(train_loader, net, optimizer, True, epoch)
    test_loss, test_accuracy = one_loop(test_loader, net, optimizer, False, epoch)

    train_losses.append(train_loss)
    test_losses.append(test_losses)

    run.log({"train/loss": train_loss, "train/acc": train_accuracy, "test/loss": test_loss, "test/acc": test_accuracy})

    save_path = save_base_path / f"ckpt/epoch-{epoch}.pkl"
    torch.save({"model": net.state_dict(), "optim": optimizer.state_dict()}, save_path)

    print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")


# fig, ax = plt.subplots(1, 2)

# x = np.arange(nb_epoch)
# plt.plot(x, train_losses)
# plt.plot(x, test_losses)
# plt.savefig("base_training_results.png", dpi=500)
# plt.show()
# ax[0].plot(x, train_losses)
# ax[1].plot(x, test_losses)
# fig.savefig("base_training_results.png", dpi=500)
