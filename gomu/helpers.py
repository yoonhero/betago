import os
import functools
import time

class GameInfo:
    def __init__(self, nrow, ncol, n_to_win):
        self.nrow = nrow
        self.ncol = ncol
        self.n_to_win = n_to_win
    def __call__(self):
        # return {"nrow": self.nrow, "ncol": self.ncol, "n_to_win": self.n_to_win}
        return [self.nrow, self.ncol, self.n_to_win]

class DEBUGSign():
    value: int
    def __init__(self, default): self.value = type(default)(os.getenv("DEBUG", default))
    def __bool__(self): return bool(self.value)
    def __gt__(self, x): return self.value > x
    def __ge__(self, x): return self.value >= x
    def __le__(self, x): return self.value <= x
    def __lt__(self, x): return self.value < x

DEBUG = DEBUGSign(0)

def loading(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        st = time.monotonic()
        result = func(*args, **kwargs)
        end = time.monotonic()
        if DEBUG >= 2:
            print(f"Loading the PolicyValue Network in {end - st}s")
        return result
    return wrapped

models_channels = {
    "old": [2, 64, 128, 256, 128, 64, 32, 1],
    "new": [2, 64, 128, 64, 32, 1]
}

base_game_info = GameInfo(20, 20, 5)