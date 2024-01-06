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

from gomu.base import PolicyValueNet, NewPolicyValueNet

is_gui = os.getenv("GUI")
device = os.getenv("DEVICE", "mps") # cuda | cpu | mps
bot_type = os.getenv("BOT", "random") # random | torch | minimax | dijkstra | qstar
max_vertex = int(os.getenv("MAX_VERTEX", 3))
max_depth = int(os.getenv("MAX_DEPTH", 3))
first_channel = int(os.getenv("FIRST_CHAN", 2)) # default is 2 | history_size + 2

# Gomu Board
nrow = 3
ncol = 3
n_to_win = 3
game_info = GameInfo(nrow=nrow, ncol=ncol, n_to_win=n_to_win)

# Load Agent
ckp_path = os.getenv("LOAD", "./models/1224-256.pkl")
first_channel = int(os.getenv("FIRST_CHAN", 2))
with_history = first_channel != 2
model = load_base(game_info=game_info, device=device, ckp_path=ckp_path)

base_config = {"model": model, "device": device, "n_to_win":n_to_win, "with_history": with_history}

if "random" in bot_type:
    bot = RandomMover(n_to_win=n_to_win)
if "torch" in bot_type:
    bot = PytorchAgent(**base_config)
if "minimax" in bot_type:
    bot = MinimaxWithAB(**base_config, max_search_vertex=max_vertex, max_depth=max_depth)
if "dijkstra" in bot_type:
    bot = DijkstraAgent(**base_config, max_depth_per_search=max_depth, max_search_vertex=max_vertex)
if "qstar" in bot_type:
    bot = QstarAgent(**base_config, max_depth=max_depth, max_vertexs=max_vertex)

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
    game = GomokuGUI(rows=nrow, cols=ncol, n_to_win=n_to_win, bot=bot, size=100)
    game.play()