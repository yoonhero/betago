import numpy as np
from argparse import ArgumentParser

from gomu import GoMuKuBoard
from gomu import Nerd
from gomu.gui import GomokuGUI

parser = ArgumentParser()
parser.add_argument("--gui", action="store_true")

args = parser.parse_args()
is_gui = args.gui

nrow = 5
ncol = 5

board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=1)
bot = Nerd(-1, 5)


if not is_gui:
    while True:
        inp = input("> YOU: x, y ")
        x, y = [int(pos) for pos in inp.split(",")]
        # if player set a stone in not available space.
        if not board.set(x, y):
            print("Please retry.")
            continue

        print("Bot Thinking...")
        freeee = np.argwhere(board.board==0).tolist()
        selected_pos = bot(board.board, freeee)
        selected_x, selected_y = selected_pos
        board.set(selected_x, selected_y)
        print("Bot Thinking was completed.")
        print(board)
else:
    game = GomokuGUI(rows=5, cols=5, n_to_win=3, bot=bot)
    game.play()