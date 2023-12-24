#! /usr/bin/env python
import numpy as np
import os
import torch
import time

from gomu.helpers import DEBUG
from gomu.gomuku import GoMuKuBoard
from gomu.gomuku.gui import GomokuGUI
from gomu.bot import *
from gomu.algorithms import *

from gomu.base import PocliyValueNet

is_gui = os.getenv("GUI")
# is_gui = True
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps"
bot_type = os.getenv("BOT", "random")
max_vertex = int(os.getenv("MAX_VERTEX", 3))
max_depth = int(os.getenv("MAX_DEPTH", 3))

# Gomu Board
nrow = 20
ncol = 20
n_to_win = 5

# Load Agent
cpk_path = os.getenv("LOAD", "./models/1217-256.pkl")
if device == "cpu" or device == "mps":
    checkpoint = torch.load(cpk_path, map_location=torch.device(device))["model"]
else: checkpoint = torch.load(cpk_path)["model"]
channels = [2, 64, 128, 256, 128, 64, 32, 1]
start = time.time()
model = PocliyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=0.0)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

if DEBUG>=2:
    print(f"Loading the PolicyValue Network in {time.time() - start}s")

base_config = {"model": model, "device": device, "n_to_win":n_to_win}

if "random" in bot_type:
    bot = RandomMover(n_to_win=n_to_win)
if "torch" in bot_type:
    bot = PytorchAgent(**base_config)
if "minimax" in bot_type:
    bot = MinimaxWithAB(**base_config, max_search_vertex=max_vertex, max_depth=max_depth)
if "dijkstra" in bot_type:
    bot = DijkstraAgent(**base_config, max_depth_per_search=max_depth, max_search_vertex=max_vertex)

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
        selected_poses, _ = bot(board_state)
        
        col, row = selected_poses
        if not board.set(col, row):
            print("Bot Error!!!")
            break
        print("Bot Thinking was completed.")
        print(board)
else:
    game = GomokuGUI(rows=nrow, cols=ncol, n_to_win=n_to_win, bot=bot, size=45)
    game.play()