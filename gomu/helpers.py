import os

class GameInfo:
    def __init__(self, nrow, ncol, n_to_win):
        self.nrow = nrow
        self.ncol = ncol
        self.n_to_win = n_to_win

class DEBUGSign():
    value: int
    def __init__(self, default): self.value = type(default)(os.getenv("DEBUG", default))
    def __bool__(self): return bool(self.value)
    def __gt__(self, x): return self.value > x
    def __ge__(self, x): return self.value >= x
    def __lt__(self, x): return self.value < x

DEBUG = DEBUGSign(0)