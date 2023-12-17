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
    def __init__(self, n_to_win):
        self.n_to_win = n_to_win
        self.turn = None

    def set_turn(self, turn):
        self.turn = turn

    def forward(self, state, free_spaces):
        return NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    # Validate the predicted zone was possible or not.
    def validate(self, predicted, free_spaces):
        return filter(predicted, lambda pos: pos not in free_spaces)



# Actually just stupid guy
class RandomMover(Agent):
    def forward(self, state, free_spaces):
        return random.choice(free_spaces)



# Simple Pytorch Agent
class PytorchAgent(Agent):
    def __init__(self, model, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device

   # def preprocess_state(self, state):
    #    torch_state = torch.from_numpy(state)
     #   nrow, ncol = torch_state.shape[-2], torch_state.shape[-1]
      #  torch_state = torch_state.view(-1, 2, nrow, ncol)
       # return torch_state

    def process_free_spaces(self, batch_free_spaces, shape):
        batch, nrow, ncol = shape
        tmp = torch.zeros((batch, nrow*ncol), dtype=torch.long)
        for b, free_spaces in enumerate(batch_free_spaces):
            for pos in free_spaces:
                idx = pos[0] + pos[1] * ncol 
                tmp[b, idx] = 1
        return tmp

    @torch.no_grad()
    def model_predict(self, state):
        if isinstance(state, np.ndarray):
            #state = self.preprocess_state(state)
            state = torch.from_numpy(state).to(dtype=torch.float32, device=self.device)
        
        policy, value = self.model(state)
        return policy, value
    
    def format_pos(self, indices, ncol):
        return [(idx%ncol, idx//ncol) for idx in indices][0]
    
    def predict_next_pos(self, board_state, not_free_space, top_k):
        # processed_batch_free_spaces = self.process_free_spaces(batch_free_spaces, batch_state.shape)
        policy, value = self.model_predict(board_state)

        not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32, device=self.device)
        not_possible[not_possible==1] = -float("inf")
        policy += not_possible
        policy = policy.view(not_free_space.shape[0], -1).softmax(-1)
        topk_policy_values, topk_policy_indices = torch.topk(policy, top_k, -1)
        selected_policy = torch.multinomial(topk_policy_values, num_samples=1)
        policies = torch.gather(topk_policy_indices, 1, selected_policy)
        predicted_pos = [self.format_pos(batch, ncol=board_state.shape[-1]) for batch in policies.tolist()]
        return predicted_pos, value

    def forward(self, state, not_free_spaces, top_k=3):
        return self.predict_next_pos(state, not_free_spaces, top_k=top_k)


class Minimax(PytorchAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    

