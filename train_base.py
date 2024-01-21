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
import datetime

import multiprocessing
import wandb
from pathlib import Path
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from essential import exp_logger

from gomu.helpers import DEBUG
from gomu.base import get_total_parameters, NewPolicyValueNet
from gomu.viz import tensor2gomuboard

from gomu.data_utils import GOMUDataset

def get_loss(policy, value, GT, win, nrow, ncol, mask=None, exploration_rate=0.001, eps=1e-10):
    # cross_loss = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    # bce = nn.BCELoss()

    ypred = policy.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    ypred = ypred.softmax(1)
    y = GT.permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol)
    # y /= y.sum(-1, keepdim=True)
    # cross_en_loss = -(y * torch.log(ypred)).sum(1).mean()
    logistic_loss = -(y*torch.log(ypred+eps) + (1-y)*torch.log(1-ypred+eps))
    mse_loss = mse(ypred, y)
    entropy_loss = (ypred * torch.log(ypred+eps))
    if mask != None:
        logistic_loss *= mask
        entropy_loss *= mask
    # Boost the uniform.
    logistic_loss = logistic_loss.sum(-1).mean()
    entropy_loss = entropy_loss.sum(-1).mean()

    # Value MSE
    value_loss = mse(value, win)

    return logistic_loss+mse_loss+exploration_rate*entropy_loss, value_loss

def save_result(x, y, policy, nrow, ncol, epoch, logger, train=True):
    pred_pos_pil = tensor2gomuboard(policy, nrow, ncol, softmax=True, scale=10)
    concatenated = x[0]-x[-1]
    ground_true_pil = tensor2gomuboard(2*(y!=0)+concatenated, nrow, ncol)
    if train:
        tag = "trainresults"
    else:
        tag = "evalresults"
    logger.save_images(tag, [pred_pos_pil, ground_true_pil], epoch, ["pred", "gt"])

def training_one_epoch(loader, net, optimizer, training, epoch, nrow, ncol, **kwargs):
    _loss = []
    num_correct = 0
    num_samples = 0

    if training:
        with tqdm.tqdm(loader) as pbar:
            for step, (X, Y, win) in enumerate(pbar):
                net.train()
                # Mask to prevent dumb learning a something.
                mask = (X.sum(1, keepdim=True)==0).permute(0, 2, 3, 1).contiguous().view(-1, nrow*ncol).detach()
                policy, value = net(X)
                pi_loss, value_loss = get_loss(policy, value, Y, win, nrow, ncol, mask=mask)
                loss = pi_loss+value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _loss.append(loss.item()) 

                pbar.set_description(f"epoch: {epoch} step: {step} pi: {pi_loss.item()} value: {value_loss.item()}")
                # pbar.set_postfix(loss=_loss[-1])
                
                num_correct += ((value>0)==((win+1)/2)).sum()
                num_samples += value.size(0)
            
            save_result(X[0], Y[0], policy[0], nrow=nrow, ncol=ncol, epoch=epoch, **kwargs)

    if not training:
        with torch.no_grad():
            net.eval()
            for (X, Y, win) in tqdm.tqdm(loader, desc="Testing..."):
                policy, value = net(X)
                pi_loss, value_loss = get_loss(policy, value, Y, win, nrow, ncol)
                _loss.append((pi_loss+value_loss).item())

                num_correct += ((value>0)==((win+1)/2)).sum()
                num_samples += value.size(0)

            save_result(X[0], Y[0], policy[0], nrow=nrow, ncol=ncol, train=False, epoch=epoch, **kwargs)

    return sum(_loss) / len(_loss), num_correct / num_samples



if __name__ == "__main__":
    resuming = bool(int(os.getenv("RESUME", "0")))
    load_path = os.getenv("CKPT")
    save_term = int(os.getenv("SAVE_TERM", 1))
    device = os.getenv("DEVICE", "mps")
    save_dir = os.getenv("SAVE", "./tmp")
    log = bool(int(os.getenv("LOG", 1)))

    # total_samples = 14000
    total_samples = 100
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    full_dataset = GOMUDataset(total_samples, device=device)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # batch_size = 256
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    nrow, ncol = 20, 20
    # channels = [2, 8, 36]
    channels = [2, 64, 128, 64, 1]
    #net = Unet(nrow=nrow, ncol=ncol, channels=channels).to(device)
    dropout = 0.5
    net = NewPolicyValueNet(nrow, ncol, channels, dropout=dropout).to(device)

    learning_rate = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_parameters = get_total_parameters(net)
    print(total_parameters)

    # Resuming the training.
    if resuming:
        checkpoint = torch.load(load_path, map_location=device)
        net.load_state_dict(state_dict=checkpoint["model"])
        optimizer.load_state_dict(state_dict=checkpoint["optim"])

    # value = datetime.datetime.fromtimestamp(time.time())
    # save_base_folder_name = value.strftime('%Y%m%d-%H%M%S')
    # save_base_path = Path(f"{save_dir}/history_{save_base_folder_name}")
    # save_base_path.mkdir(exist_ok=True)
    # (save_base_path / "ckpt").mkdir(exist_ok=True)
    # (save_base_path / "evalresults").mkdir(exist_ok=True)
    # (save_base_path / "trainresults").mkdir(exist_ok=True)

    model_cfg = {"channels": channels, "nrow": nrow, "ncol": ncol, "param": total_parameters, "dropout": dropout}
    exp_cfg = {"learning_rate": learning_rate, "train_size": train_size, "test_size": test_size, "total_samples": total_samples}
    configs = {**model_cfg, **exp_cfg}
        
    if log:
        mode = exp_logger.ONLINE
    else: mode = exp_logger.OFFLINE
    run = exp_logger.Logger(mode, run_name="AlphaGomu", configs=configs)

    train_losses = []
    test_losses = []
    nb_epoch = 50
    for epoch in range(nb_epoch):
        train_loss, train_accuracy= training_one_epoch(train_loader, net, optimizer, True, epoch, nrow=nrow, ncol=ncol, logger=run)
        test_loss, test_accuracy = training_one_epoch(test_loader, net, optimizer, False, epoch, nrow=nrow, ncol=ncol, logger=run)

        train_losses.append(train_loss)
        test_losses.append(test_losses)

        run.log({"train/loss": train_loss, "train/acc": train_accuracy, "test/loss": test_loss, "test/acc": test_accuracy}, epoch)

        if (epoch+1) % save_term == 0:
            run.log_ckp(net, optimizer, f"epoch-{epoch}.pkl")

        if DEBUG >= 1:
            print(f"{epoch}th EPOCH DONE!! ---- Train: {train_loss} | Test: {test_loss}")