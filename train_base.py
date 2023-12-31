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
import os

from gomu.base import PolicyValueNet, Unet, Transformer, get_total_parameters
from gomu.viz import tensor2gomuboard

class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.base_path = Path("./dataset/processed")
        self.board_state, self.next_pos, self.winner, self.prev_moves = self._load(max_idx)
        
        self.total = self.board_state.shape[0]
        # self.total = max_idx

    def _load(self, max_idx):
        inputs = []
        outputs = []
        game_results = []
        prev_moves = []

        for i in tqdm.tqdm(range(max_idx+1)):
            # inputs, outputs = p.map(self._load_file, list(range(max_idx+1)))
            _inputs, _outputs, _results, _prev_move = self._load_file(i)
            inputs.append(_inputs)
            outputs.append(_outputs)
            game_results.append(_results)
            prev_moves.append(_prev_move)
                
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)
        game_results = np.concatenate(game_results)
        prev_moves = np.concatenate(prev_moves)
        
        return inputs, outputs, game_results, prev_moves
   
    def _load_file(self, i):
        file_path = self.base_path / f"{i:05d}.npz"
        data = np.load(file_path, allow_pickle=True)
        
        _inputs = data["inputs"]
        _outputs = data["outputs"]
        _results = data["results"]
        _prev_moves = data["prev_pos"]

        return _inputs, _outputs, _results, _prev_moves

    # Return [Board, NextPos, Win%]
    def __getitem__(self, idx):
        # x, y = self._load_file(idx)
        # randomnum = random.randint(0, 12)
        # to_get_idx = idx*12+randomnum
        board, next_pos, game_result, prev_move = self.board_state[idx], self.next_pos[idx], self.winner[idx], self.prev_moves[idx]
        # BLACK: 1, WHITE: -1: DRAW=0
        x, y = torch.from_numpy(board), torch.from_numpy(next_pos)
        
        # Masking the y
        # not_free_space = x.sum(1).unsqueeze(1)
        # y = y.masked_fill(not_free_space==1, -1e9)

        # if your turn?
        # me in first row
        # comptetitor in second row
        total_BLACK = torch.sum(x[0])
        total_WHITE = torch.sum(x[1])
        if total_BLACK != total_WHITE:
            # print(total_BLACK, total_WHITE)
            # x[0], x[1] = x[1], x[0]
            x = torch.roll(x, 1, 0)
            game_result = -game_result

        # x = torch.cat([x, prev_move])

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

def get_loss(policy, value, GT, win):
    cross_loss = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    ypred = policy.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    # flatten_pred = einops.rearrange(policy, "b c h w -> b (c h w)")
    # y = einops.rearrange(GT, "b c h w -> b (c h w)")
    y = GT.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    y = y.argmax(-1)
    cross_en_loss = cross_loss(ypred, y)
    # mse_loss = mse(policy, GT)
    # policy_loss = alpha*cross_en_loss + (1-alpha)*mse_loss
    policy_loss = cross_en_loss
    value_loss = bce(value, win)

    return policy_loss+value_loss

def training_one_epoch(loader, net, optimizer, training, epoch):
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



if __name__ == "__main__":
    resuming = bool(int(os.getenv("RESUME", "0")))
    load_path = os.getenv("CKPT")
    save_term = int(os.getenv("SAVE_TERM", 1))

    total_samples = 14000
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    full_dataset = GOMUDataset(total_samples, device=device)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # batch_size = 256
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    nrow, ncol = 20, 20
    # channels = [2, 8, 36]
    channels = [2, 64, 128, 256, 128, 64, 32, 1]
    #net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
    dropout = 0.5
    net = PolicyValueNet(nrow, ncol, channels, dropout=dropout).to(device)

    # net = torch.compile(not_compiled)

    # net = Transformer(1, 1, (20, 20), 16, 64).to(device)
    # learning_rate = 0.001
    learning_rate = 0.001
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.1)
    total_parameters = get_total_parameters(net)
    print(total_parameters)

    # Resuming the training.
    if resuming:
        checkpoint = torch.load(load_path, map_location=device)
        net.load_state_dict(state_dict=checkpoint["model"])
        optimizer.load_state_dict(state_dict=checkpoint["optim"])

    save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
    save_base_path.mkdir(exist_ok=True)
    (save_base_path / "ckpt").mkdir(exist_ok=True)
    (save_base_path / "evalresults").mkdir(exist_ok=True)

    grad_clip=2

    model_cfg = {"channels": channels, "nrow": nrow, "ncol": ncol, "param": total_parameters, "dropout": dropout}
    exp_cfg = {"learning_rate": learning_rate, "train_size": train_size, "test_size": test_size, "grad_clip": grad_clip, "total_samples": total_samples}
        
    run = wandb.init(
        project="AlphaGomu",
        config={**model_cfg, **exp_cfg}
    )

    train_losses = []
    test_losses = []
    nb_epoch = 50
    for epoch in range(nb_epoch):
        train_loss, train_accuracy= training_one_epoch(train_loader, net, optimizer, True, epoch)
        test_loss, test_accuracy = training_one_epoch(test_loader, net, optimizer, False, epoch)

        train_losses.append(train_loss)
        test_losses.append(test_losses)

        run.log({"train/loss": train_loss, "train/acc": train_accuracy, "test/loss": test_loss, "test/acc": test_accuracy})

        if (epoch+1) % save_term == 0:
            save_path = save_base_path / f"ckpt/epoch-{epoch}.pkl"
            torch.save({"model": net.state_dict(), "optim": optimizer.state_dict()}, save_path)

        print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")