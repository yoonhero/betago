# MCTS Alpha-Zero Implementatino on GO.
import uuid
from queue import Queue
import os
from collections import defaultdict
import math
import random

import torch
import numpy as np

from gomu.gomuku import board
from gomu.base import PolicyValueNet
from gomu.bot import *
from gomu.game_tree import Graph

from train_base import get_loss

max_search = int(os.environ("MAX", 10))

nrow = 10
ncol = 10
n_to_win = 5
channels = [2, 64, 128, 256, 128, 64, 32, 1]
dropout = 0.5

device = "mps"

model = PolicyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=dropout)
model.to(device)

agent = PytorchAgent(model=model, device=device, n_to_win=n_to_win)

history = Queue()


class MCTSNode():
    def __init__(self, state, turn, parent=None):
        self.state = state
        self.parent = parent
        self.childrens = []
        self.action = None

        self.value_sum = 0
        self.visit_count = 0

        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0

        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.id = uuid.uuid4()

        self.turn = turn

    def untried_actions(self):
        _untried_actions = self.get_legal_actions()
        return _untried_actions

    def get_legal_actions(self):
        next_poses, _ = agent.predict_next_pos(board_state=self.state, top_k=5)
        return next_poses

    def q(self):
        win = self._results[1]
        lose = self._results[-1]
        return win - lose

    def n(self):
        return self.visit_count
    
    def expand(self):
        action = self._untried_actions.pop()
        next_state = agent.get_new_board_state(board_state=self.state, next_pos=action, my_turn=self.turn)
        child_node = MCTSNode(next_state, turn=1-self.turn, parent=self)
        self.children.append(child_node)

        return child_node

    def is_terminal_node(self):
        return self.game_result() != 0

    def game_result(self):
        is_game_done = GoMuKuBoard.is_game_done(board_state=self.state, turn=self.turn, n_to_win=n_to_win)
        if is_game_done:
            return 1
        return 0

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def rollout(self):
        current_rollout_state = self.state
        turn = self.turn

        while not GoMuKuBoard.is_game_done(board_state=current_rollout_state, turn=turn, n_to_win=self.n_to_win):
            turn = 1 - turn
            possible_moves, _ = agent.predict_next_pos(board_state=current_rollout_state, top_k=5)

            action = self.rollout_policy(possible_moves)
            current_rollout_state = agent.get_new_board_state(board_state=current_rollout_state, next_pos=action, my_turn=turn)
            
        return int(GoMuKuBoard.is_game_done(board_state=current_rollout_state), turn=turn, n_to_win=self.n_to_win)

    # def rollout(self)
    def backpropagate(self, result):
        self.visit_count += 1
        self._results[result] += 1

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return not int(self._untried_actions.__len__())

    def score(self, c=0.1):
        ucb_score = self.q() / self.n() + c * math.sqrt(2 * math.log(self.parent.n()) / self.n())
        return ucb_score
    
    def best_child(self, c_param):
        choices_weight = [children.score(c=c_param) for children in self.childrens]
        return self.children[np.argmax(choices_weight)]
    
    def _tree_policy(self):
        current_node = self

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                current_node.expand()
            else:
                current_node = current_node.best_child()
        
        return current_node
    
    def best_action(self):
        simulation_no = 100
        
        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
    
        return self.best_child(c_param=0.)

    def __repr__(self) -> str:
        return f"MCTS(value={self.value()}, pos={self.next_pos})"


def ucb_score():
    return None


def one_cycle(tree):
    count = 0
    current_board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win)
    while count <= max_search:
        
        count += 1
