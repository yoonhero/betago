import os

class DEBUGSign():
    def __init__(self): self.value = os.getenv("DEBUG") or 1
    def set(self, x): self.value = x
    def __gt__(self, x): return self.value < x
    def __ge__(self, x): return self.value <= x
    def __eq__(self, x): return self.value == x 

DEBUG = int(os.getenv("DEBUG"))