#! /usr/bin/env python
import numpy as np
import os
import torch
import time

from gomu.helpers import DEBUG, Games
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
module_type = os.getenv("MOD", "new")
game_type = os.getenv("TYPE", "small")

# Gomu Board
# nrow = 7
# ncol = 7
# n_to_win = 5
game_info = Games[game_type]
nrow, ncol, n_to_win = game_info()

# Load Agent
ckp_path = os.getenv("CKP", "./models/1224-256.pkl")
with_history = first_channel != 2

if module_type == "old":
    module = PolicyValueNet
else:
    module = NewPolicyValueNet

model = load_base(game_info=game_info, device=device, ckp_path=ckp_path, module=module, channels=[2, 64, 128, 256, 128, 64, 1])

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
    game = GomokuGUI(rows=nrow, cols=ncol, n_to_win=n_to_win, bot=bot, size=45)
    game.play()
