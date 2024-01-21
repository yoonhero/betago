#! /usr/bin/env python
import numpy as np
import time
import tqdm
import os

from gomu.base import NewPolicyValueNet
from gomu.helpers import DEBUG, GameInfo, base_game_info
from gomu.gomuku import GoMuKuBoard
from gomu.bot import load_base, PytorchAgent, Agent
from gomu.algorithms import *

def simulate(agent1, agent2, game_info: GameInfo, first=None):
    turn = 0
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win
    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win)

    while True:      
        if not bool(turn):
            agent: PytorchAgent = agent1
        else:
            agent: PytorchAgent = agent2

        board_state = board.board
        if first != None and int(not first) - turn == 0:
            board_state = board.get_padded_board(20)
            attention_mask = torch.zeros((1, 20, 20))
            attention_mask[:, :nrow, :ncol] = 1
            attention_mask = attention_mask.view(1, -1)
        else: attention_mask = None
        next_pos, _ = agent(board_state, turn=turn, attention_mask=attention_mask)
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
def ELO(challenger_elo, critic_elo, challenger, total_play, game_info: GameInfo, device, op_ckp="./models/1224-256.pkl", k=16, padded=False):
    nrow, ncol, n_to_win = game_info.nrow, game_info.ncol, game_info.n_to_win

    result = 0

    base = load_base(game_info=base_game_info, device=device, ckp_path=op_ckp)
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

        if padded:
            game_result = simulate(**agent, game_info=game_info)
        else:
            game_result = simulate(**agent, game_info=game_info, first=not bool(challenger_turn))

        result += game_result[1-challenger_turn]

    p_model_win = 1 / (1+pow(10,(critic_elo-challenger_elo)/400))
    elo = challenger_elo + k * (result - p_model_win * total_play)

    if DEBUG>=1:
        print(f"Model Expected Winning P: {p_model_win} | Real Winning P: {result / total_play}")        
        print(f"Final ELO {elo}")

    return elo
    

if __name__ == "__main__":
    device = "cpu"
    model_path = os.getenv("CKP", "./models/1224-256.pkl")
    game_info = GameInfo(nrow=20, ncol=20, n_to_win=5)
    channels = [2, 64, 128, 64, 1]
    model = load_base(game_info=game_info, device=device, module=NewPolicyValueNet, ckp_path=model_path, channels=channels)
    model = PytorchAgent(model, device=device, n_to_win=game_info.n_to_win, with_history=False)

    import matplotlib.pyplot as plt

    elos = {"base": 500, "upcoming": 500}
    elos["upcoming"] = ELO(elos["upcoming"], elos["base"], model, 100, game_info=game_info, device=device)
    print(f"Final Result! : {elos}")
    plt.bar(list(elos.keys()), list(elos.values()))
    plt.ylabel("Elo")
    # plt.savefig("elo.png")
    plt.show()