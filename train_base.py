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

from gomu.base import PolicyValueNet, Unet, Transformer, get_total_parameters, NewPolicyValueNet
from gomu.viz import tensor2gomuboard

from gomu.data_utils import GOMUDataset

def get_loss(policy, value, GT, win, nrow, ncol, exploration_rate=0.05):
    # cross_loss = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    # bce = nn.BCELoss()

    ypred = policy.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    ypred = ypred.softmax(1)
    y = GT.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    cross_en_loss = -(y * torch.log(ypred)).sum(1).mean()
    # Boost the uniform.
    # entropy_loss = -(ypred * torch.log(ypred)).sum(1).mean()
    policy_loss = cross_en_loss

    # Value MSE
    value_loss = mse(value, win)

    return policy_loss+value_loss

def save_result(x, y, policy, save_base_path, nrow, ncol, epoch, train=True):
    pred_pos_pil = tensor2gomuboard(policy, nrow, ncol, softmax=True, scale=10)
    concatenated = x[0]-x[2]
    ground_true_pil = tensor2gomuboard(2*(y!=0)+concatenated, nrow, ncol)
    if train:
        eval_result_path = save_base_path / f"trainresults/{epoch}-pred.png"
        eval_gt_path = save_base_path / f"trainresults/{epoch}-gt.png"
    else:
        eval_result_path = save_base_path / f"evalresults/{epoch}-pred.png"
        eval_gt_path = save_base_path / f"evalresults/{epoch}-gt.png"
    pred_pos_pil.save(eval_result_path)
    ground_true_pil.save(eval_gt_path)

def training_one_epoch(loader, net, optimizer, training, epoch, nrow, ncol, save_base_path):
    _loss = []
    num_correct = 0
    num_samples = 0

    if training:
        with tqdm.tqdm(loader) as pbar:
            for step, (X, Y, win) in enumerate(pbar):
                net.train()
                policy, value = net(X)
                loss = get_loss(policy, value, Y, win, nrow, ncol)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _loss.append(loss.item()) 

                pbar.set_description(f"{epoch}/{step}")
                pbar.set_postfix(loss=_loss[-1])
                
                num_correct += ((value>0)==win).sum()
                num_samples += value.size(0)
            
            save_result(X[0], Y[0], policy[0], save_base_path, nrow=nrow, ncol=ncol, epoch=epoch)

    if not training:
        with torch.no_grad():
            net.eval()
            for (X, Y, win) in tqdm.tqdm(loader, desc="Testing..."):
                policy, value = net(X)
                output = get_loss(policy, value, Y, win, nrow, ncol)
                _loss.append(output.item())

                num_correct += ((value>0.5)==win).sum()
                num_samples += value.size(0)

            save_result(X[0], Y[0], policy[0], save_base_path, nrow=nrow, ncol=ncol, train=False, epoch=epoch)

    return sum(_loss) / len(_loss), num_correct / num_samples



if __name__ == "__main__":
    resuming = bool(int(os.getenv("RESUME", "0")))
    load_path = os.getenv("CKPT")
    save_term = int(os.getenv("SAVE_TERM", 1))
    device = os.getenv("DEVICE", "mps")
    save_dir = os.getenv("SAVE", "./tmp")

    total_samples = 10000
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    full_dataset = GOMUDataset(total_samples, device=device)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # batch_size = 256
    batch_size = 4086
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    nrow, ncol = 20, 20
    # channels = [2, 8, 36]
    channels = [3, 64, 128, 256, 128, 64, 1]
    #net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
    dropout = 0.2
    net = NewPolicyValueNet(nrow, ncol, channels, dropout=dropout).to(device)

    # net = torch.compile(not_compiled)

    # net = Transformer(1, 1, (20, 20), 16, 64).to(device)
    # learning_rate = 0.001
    learning_rate = 0.001
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_parameters = get_total_parameters(net)
    print(total_parameters)

    # Resuming the training.
    if resuming:
        checkpoint = torch.load(load_path, map_location=device)
        net.load_state_dict(state_dict=checkpoint["model"])
        optimizer.load_state_dict(state_dict=checkpoint["optim"])

    save_base_path = Path(f"{save_dir}/history_{int(time.time()*1000)}")
    save_base_path.mkdir(exist_ok=True)
    (save_base_path / "ckpt").mkdir(exist_ok=True)
    (save_base_path / "evalresults").mkdir(exist_ok=True)
    (save_base_path / "trainresults").mkdir(exist_ok=True)

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
        train_loss, train_accuracy= training_one_epoch(train_loader, net, optimizer, True, epoch, nrow=nrow, ncol=ncol, save_base_path=save_base_path)
        test_loss, test_accuracy = training_one_epoch(test_loader, net, optimizer, False, epoch, nrow=nrow, ncol=ncol, save_base_path=save_base_path)

        train_losses.append(train_loss)
        test_losses.append(test_losses)

        run.log({"train/loss": train_loss, "train/acc": train_accuracy, "test/loss": test_loss, "test/acc": test_accuracy})

        if (epoch+1) % save_term == 0:
            save_path = save_base_path / f"ckpt/epoch-{epoch}.pkl"
            torch.save({"model": net.state_dict(), "optim": optimizer.state_dict()}, save_path)

        print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")