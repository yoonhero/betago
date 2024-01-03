#! /usr/bin/env python
import numpy as np
import time
import tqdm

from gomu.helpers import DEBUG, GameInfo
from gomu.gomuku import GoMuKuBoard
from gomu.bot import load_base, PytorchAgent
from gomu.algorithms import *


def ELO(model_elo, base_elo, model, total_play, game_info: GameInfo, device, k=16):
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win

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

            # Stochastical Variation.
            next_pos, value = agent.predict_next_pos(board.board, top_k=3, best=True)
            col, row = next_pos[0]
            board.set(col, row)
            
            base_agent.update_history(next_pos)
            model_agent.update_history(next_pos)
            turn = 1 - turn

            if board.is_gameover():
                if models_turn == turn:
                    result += 1
                if DEBUG >= 1:
                    GoMuKuBoard.viz(board.board).show()
                break
            if board.is_draw():
                result += 0.5
                break

    p_model_win = 1 / (1+pow(10,(base_elo-model_elo)/400))
    elo = model_elo + k * (result - p_model_win * total_play)

    if DEBUG>=1:
        print(f"Model Expected Winning P: {p_model_win} | Real Winning P: {result / total_play}")        
        print(f"Final ELO {elo}")

    return elo
    

if __name__ == "__main__":
    device = "cpu"
    game_info = GameInfo(nrow=20, ncol=20, n_to_win=5)
    base_model = load_base(game_info=game_info, device=device, ckp_path="./models/")

    print(ELO(500, 500, base_model, 20, game_info=game_info, device=device))
    