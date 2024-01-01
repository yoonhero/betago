#! /usr/bin/env python
import numpy as np
import time
import tqdm

from gomu.helpers import DEBUG, GameInfo
from gomu.gomuku import GoMuKuBoard
from gomu.bot import load_base, PytorchAgent
from gomu.algorithms import *


def ELO(model, total_play, game_info: GameInfo, device):
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win

    model_elo = 100
    base_elo = 500
    k = 16

    result = 0

    base = load_base(game_info=game_info, device=device)
    base_agent = PytorchAgent(base, device=device, n_to_win=n_to_win, with_history=False)
    model_agent = PytorchAgent(model, device=device, n_to_win=n_to_win, with_history=False)

    for i in tqdm.trange(total_play):
        board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win)
        turn = 0
        models_turn = i % 2

        while True:      
            if models_turn == turn:
                agent: PytorchAgent = model_agent
            else:
                agent: PytorchAgent = base_agent

            next_pos, value = agent(board.board)
            board.set(next_pos)
            
            base_agent.update_history(next_pos)
            model_agent.update_history(next_pos)
            turn = 1 - turn

            if board.is_gameover():
                if models_turn == turn:
                    result += 1
            if board.is_draw():
                result += 0.5
        
    p_model_win = 1 / (1+10^((base_elo-model_elo)/400))

    return model_elo + k * (result - p_model_win * total_play)
    