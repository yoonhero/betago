from typing import Any
import numpy as np
import random
from scipy.signal import convolve2d

import torch
import torch.nn as nn
from einops import rearrange

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

    def forward(self, state, free_spaces):
        return NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    # Validate the predicted zone was possible or not.
    def validate(self, predicted, free_spaces):
        return filter(predicted, lambda pos: pos not in free_spaces)

# MCTS implementation
class PytorchAgentHeritage(Agent):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def preprocess_state(self, state):
        torch_state = torch.from_numpy(state)
        nrow, ncol = torch_state.shape[-2], torch_state.shape[-1]
        torch_state = torch_state.view(-1, nrow*ncol)

        return torch_state

    def process_free_spaces(self, batch_free_spaces, shape):
        batch, nrow, ncol = shape
        tmp = torch.zeros((batch, nrow*ncol), dtype=torch.long)
        for b, free_spaces in enumerate(batch_free_spaces):
            for pos in free_spaces:
                idx = pos[0] + pos[1] * ncol 
                tmp[b, idx] = 1
        return tmp

    def model_predict(self, state):
        if isinstance(state, np.ndarray):
            state = self.preprocess_state(state)
        pred = self.model(state)
        return pred
    
    def format_pos(self, indices, ncol):
        return [(idx//ncol, idx%ncol) for idx in indices]
    
    def predict_next_pos(self, batch_state, batch_free_spaces, top_k):
        processed_batch_free_spaces = self.process_free_spaces(batch_free_spaces, batch_state.shape)
        pred = self.model_predict(batch_state)

        not_possible = processed_batch_free_spaces * -float("inf")
        pred += not_possible
        
        predicted_pos = [self.format_pos(batch, ncol=batch_state.shape[-1]) for batch in torch.topk(pred, top_k, -1)["indices"].tolist()]
        return predicted_pos
    
    def for_singlebatch(self, state, free_spaces, top_k):
        batch_state = np.expand_dims(state, 0)
        batch_free_spaces = np.expand_dims(free_spaces, 0)
        return self.predict_next_pos(batch_state=batch_state, batch_free_spaces=batch_free_spaces)[0]

    @classmethod
    def from_trained(cls, cpk_path, model: nn.Module, **kwargs):
        checkpoint = torch.load(cpk_path)["model"]
        model = model.load_state_dict(checkpoint)
        return cls(model=model, **kwargs)


# Actually just stupid guy
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
    

class StatisticalSimlaritySearchingIntelligentGuy(Agent):
    def __init__(self, turn, n_to_win, dataset_path):
        super().__init__(turn, n_to_win)
        self.dataset = self.load_dataset(dataset_path=dataset_path)

    def load_dataset(self, dataset_path):
        return 

    def search_similar(self):
        return

    def forward(self, state, free_spaces):
        next_poses = self.search_similar(state)
        possible_poses = self.validate(next_poses, free_spaces)

        return
    


# class ItMayBeSupervisedLearningIsBetterThanReinforcementLearningButIdontThinksothisguy(Agent):
#     def __init__(self, turn, n_to_win, )

#     @classmethod
