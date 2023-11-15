from typing import Any
import numpy as np
import random
from scipy.signal import convolve2d

def convolution_calc(x, kernel1, kernel2):
    return (convolve2d(x, kernel1, mode='valid'), convolve2d(x, kernel2, mode='valid'))

def check_all_cross(tgt_board, n_to_win):
    kernel1 = np.eye(n_to_win)
    kernel2 = np.eye(n_to_win)[::-1]

    return convolution_calc(tgt_board, kernel1, kernel2)

def check_all_line(tgt_board, n_to_win):
    kernel1 = np.ones((1,n_to_win))
    kernel2 = np.ones((n_to_win, 1))

    return convolution_calc(tgt_board, kernel1, kernel2)

# Base Agent Structure
class Agent():
    def __init__(self, turn, n_to_win):
        self.turn = turn
        self.n_to_win = n_to_win

    # def is_your_turn(self, ply):

    def forward(self, state, free_spaces):
        return NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class RandomMover(Agent):
    def forward(self, state, free_spaces):
        return random.choice(free_spaces)

# Not very intelligent but wise player for proper competitors for agent
class Nerd(Agent):
    def __init__(self, turn, n_to_win):
        super().__init__(turn, n_to_win)
        dangerous_zones = []

    def checkWinningStrategy(self, state):
        zones = []

        # 0 0 0 
        # 0 1 0
        # -1 0 1
        me = state == self.turn
        not_me = state != self.turn
        
        # Diagonal
        cross, reverse_cross = check_all_cross(state, self.n_to_win)

        # line
        line, lineT = check_all_line(state, self.n_to_win)

        return zones

    def checkDangerous(self, state):
        zones = []
        return zones

    def forward(self, state, free_spaces):
        winning_zones = self.checkWinningStrategy(state)
        losing_zones = self.checkDangerous(state)

        if len(winning_zones) == 0 and len(losing_zones) == 0:
            return random.choice(free_spaces)

        return random.choice(winning_zones+losing_zones)
    