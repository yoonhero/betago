import random
from pathlib import Path
import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import einops


class GOMUDataset(Dataset):
    def __init__(self, max_idx, device):
        super().__init__()
        self.device = device
        self.base_path = Path("./dataset/processed")
        self.board_state, self.next_pos, self.winner, self.prev_moves = self._load(max_idx)
        
        self.total = self.board_state.shape[0]
        # self.total = max_idx

    def _load(self, max_idx):
        inputs = []
        outputs = []
        game_results = []
        prev_moves = []

        for i in tqdm.tqdm(range(max_idx+1)):
            # inputs, outputs = p.map(self._load_file, list(range(max_idx+1)))
            _inputs, _outputs, _results, _prev_move = self._load_file(i)
            inputs.append(_inputs)
            outputs.append(_outputs)
            game_results.append(_results)
            prev_moves.append(_prev_move)
                
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)
        game_results = np.concatenate(game_results)
        prev_moves = np.concatenate(prev_moves)
        
        return inputs, outputs, game_results, prev_moves
   
    def _load_file(self, i):
        file_path = self.base_path / f"{i:05d}.npz"
        data = np.load(file_path, allow_pickle=True)
        
        _inputs = data["inputs"]
        _outputs = data["outputs"]
        _results = data["results"]
        _prev_moves = data["prev_pos"]

        return _inputs, _outputs, _results, _prev_moves

    # Return [Board, NextPos, Win%]
    def __getitem__(self, idx):
        # x, y = self._load_file(idx)
        # randomnum = random.randint(0, 12)
        # to_get_idx = idx*12+randomnum
        board, next_pos, game_result, prev_move = self.board_state[idx], self.next_pos[idx], self.winner[idx], self.prev_moves[idx]
        # BLACK: 1, WHITE: -1: DRAW=0
        x, y = torch.from_numpy(board), torch.from_numpy(next_pos)
        
        # Masking the y
        # not_free_space = x.sum(1).unsqueeze(1)
        # y = y.masked_fill(not_free_space==1, -1e9)

        # if your turn?
        # me in first row
        # comptetitor in second row
        total_BLACK = torch.sum(x[0])
        total_WHITE = torch.sum(x[1])
        if total_BLACK != total_WHITE:
            # print(total_BLACK, total_WHITE)
            # x[0], x[1] = x[1], x[0]
            x = torch.roll(x, 1, 0)
            game_result = -game_result

        # x = torch.cat([x, prev_move])

        # WIN: 1 | DRAW: 0.5 | LOSE: 0
        game_result = (game_result+1)/2
        game_result = torch.from_numpy(game_result)

        # IMG [B, C, H, W]
       # x = einops.rearrange(x, "2 h w -> h w")
        y = einops.rearrange(y, "h w -> 1 h w")
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        game_result = game_result.to(torch.float32)

        return x.to(self.device), y.to(self.device), game_result.to(self.device)

    def __len__(self):
        return self.total

class TempDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.total = len(data)
        self.device = device
    def __getitem__(self, idx):
        state, pi, value = self.data[idx]

        total_BLACK = torch.sum(state[0])
        total_WHITE = torch.sum(state[1])
        if total_BLACK != total_WHITE:
            state = torch.roll(state, 1, 0)

        # Augmentation
        # roll? flip?
        random_event = random.choice([1, 2, 3, 4])
        state = torch.rot90(state, random_event, dims=[1, 2])
        pi = torch.rot90(pi, random_event, dims=[1, 2])

        state = state.to(torch.float32).to(self.device)
        pi = pi.to(torch.float32).to(self.device)
        value = value.to(torch.float32).to(self.device)
        return state, pi, value
    def __len__(self):
        return self.total