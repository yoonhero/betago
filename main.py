#! /usr/bin/env python
import numpy as np
from argparse import ArgumentParser
import os
import torch

from gomu.gomuku import GoMuKuBoard
from gomu.gomuku.gui import GomokuGUI
from gomu.bot import *

from gomu.base import PocliyValueNet

is_gui = os.getenv("GUI")
# is_gui = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gomu Board
nrow = 20
ncol = 20
n_to_win = 5

# Load Agent
cpk_path = "./models/1217-256.pkl"
if device == "cpu":
    checkpoint = torch.load(cpk_path, map_location=torch.device("cpu"))["model"]
else: checkpoint = torch.load(cpk_path)["model"]
channels = [2, 64, 128, 256, 128, 64, 1]
model = PocliyValueNet(nrow=nrow, ncol=ncol, channels=channels)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# bot = PytorchAgent(model=model, device=device, n_to_win=n_to_win)
# bot = RandomMover(n_to_win=n_to_win)
bot = MinimaxWithAB(model=model, device=device, n_to_win=n_to_win, max_search_node=4)


if not is_gui:
    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win)

    while True:
        print(board)
        inp = input("> YOU: x, y ")
        x, y = [int(pos) for pos in inp.split(",")]
        # if player set a stone in not available space.
        if not board.set(x, y):
            print("Please retry.")
            continue

        print("Bot Thinking...")

        # freeee = np.argwhere(board.board==0).tolist()
        # not_free_space = board.not_free_space()
        # not_free_space = np.expand_dims(not_free_space, axis=0)
        # board_state = np.expand_dims(board.board, axis=0)
        
        # print("NOT FREE", not_free_space)
        board_state = board.board
        selected_poses = bot(board_state)
        
        col, row = selected_poses[0]
        if not board.set(col, row):
            print("Bot Error!!!")
            break
        print("Bot Thinking was completed.")
        print(board)
else:
    game = GomokuGUI(rows=nrow, cols=ncol, n_to_win=n_to_win, bot=bot, size=30)
    game.play()