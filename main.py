import numpy as np
from argparse import ArgumentParser
import os

from gomu import GoMuKuBoard
from gomu.gui import GomokuGUI
from bot import *

from base import PocliyValueNet

#is_gui = os.getenv("GUI")
is_gui = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gomu Board
nrow = 20
ncol = 20
n_to_win=5


# Load Agent
cpk_path = "./tmp/history_1702722992049/ckpt/epoch-19.pkl"
checkpoint = torch.load(cpk_path)["model"]
channels = [2, 64, 128, 256, 128, 64, 1]
model = PocliyValueNet(nrow=nrow, ncol=ncol, channels=channels)
model.load_state_dict(checkpoint)
model.to(device)

#bot = PytorchAgentHeritage(model=model, device=device, turn=1, n_to_win=n_to_win)
bot = RandomMover(n_to_win=n_to_win)


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
        not_free_space = board.not_free_space()
        not_free_space = np.expand_dims(not_free_space, axis=0)
        board_state = np.expand_dims(board.board, axis=0)

        selected_poses = bot(board_state, not_free_space)[0]
        
        selected_x, selected_y = selected_poses[0]
        if not board.set(selected_x, selected_y):
            print("Bot Error!!!")
            break
        print("Bot Thinking was completed.")
        print(board)
else:
    game = GomokuGUI(rows=nrow, cols=ncol, n_to_win=n_to_win, bot=bot)
    game.play()