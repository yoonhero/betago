#! /usr/bin/env python
import numpy as np
import time
import tqdm

from gomu.helpers import DEBUG, GameInfo
from gomu.gomuku import GoMuKuBoard
from gomu.bot import load_base, PytorchAgent, Agent
from gomu.algorithms import *

def simulate(agent1, agent2, game_info: GameInfo):
    turn = 0
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win
    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win)

    while True:      
        if not bool(turn):
            agent: PytorchAgent = agent1
        else:
            agent: PytorchAgent = agent2

        # Stochastical Variation.
        next_pos, _ = agent(board.board, turn=turn)
        col, row = next_pos
        board.set(col, row)
        
        agent1.update_history(next_pos)
        agent2.update_history(next_pos)

        if board.is_gameover():
            if DEBUG >= 2:
                GoMuKuBoard.viz(board.board).show()
            
            # is Agent 1 win?
            return (1-turn, turn)
        if board.is_draw():
            return (0.5, 0.5)
        
        turn = 1 - turn

# challenger: pytorch model or agent module.
def ELO(challenger_elo, critic_elo, challenger, total_play, game_info: GameInfo, device, op_ckp="./models/1224-256.pkl", k=16):
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win

    result = 0

    base = load_base(game_info=game_info, device=device, ckp_path=op_ckp)
    critic_agent = PytorchAgent(base, device=device, n_to_win=n_to_win, with_history=False)
    if isinstance(challenger, Agent):
        challenger_agent = challenger
    else:
        challenger_agent = PytorchAgent(challenger, device=device, n_to_win=n_to_win, with_history=False)

    for i in tqdm.trange(total_play):
        challenger_turn = i % 2
        
        if not bool(challenger_turn):
            agent = {"agent1": critic_agent, "agent2": challenger_agent}
        else:
            agent = {"agent1": challenger_agent, "agent2": critic_agent}
        game_result = simulate(**agent, game_info=game_info)

        result += game_result[1-challenger_turn]

    p_model_win = 1 / (1+pow(10,(critic_elo-challenger_elo)/400))
    elo = challenger_elo + k * (result - p_model_win * total_play)

    if DEBUG>=1:
        print(f"Model Expected Winning P: {p_model_win} | Real Winning P: {result / total_play}")        
        print(f"Final ELO {elo}")

    return elo
    

if __name__ == "__main__":
    device = "cpu"
    game_info = GameInfo(nrow=20, ncol=20, n_to_win=5)
    base_model = load_base(game_info=game_info, device=device, ckp_path="./models/")

    print(ELO(500, 500, base_model, 20, game_info=game_info, device=device))