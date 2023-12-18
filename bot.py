from typing import Any
import numpy as np
import random
from scipy.signal import convolve2d
import graphviz
import uuid

import torch
import torch.nn as nn
from einops import rearrange

from helpers import DEBUG

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

    # Always first element is me~
    def preprocess_state(self, state):
        torch_state = torch.from_numpy(state).to(dtype=torch.float32, device=self.device)

        total_BLACK = state[0].sum()
        total_WHITE = state[1].sum()
        if total_BLACK != total_WHITE:
            state[0], state[1] = state[1], state[0]

        return torch_state

    @torch.no_grad()
    def model_predict(self, state):
        if isinstance(state, np.ndarray):
            state = self.preprocess_state(state).unsqueeze(0)
        policy, value = self.model(state)
        return policy, value
    
    def format_pos(self, indices, ncol):
        return [(idx%ncol, idx//ncol) for idx in indices]

    def get_not_free_space(self, board_state):
        not_free_space = board_state.sum(0)
        return not_free_space
    
    def predict_next_pos(self, board_state, top_k):
        assert top_k >= 3, "Please top_k is greater than 3."
        policy, value = self.model_predict(board_state)

        not_free_space = self.get_not_free_space(board_state=board_state)
        not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        not_possible[not_possible!=0] = -float("inf")
        policy += not_possible
        policy = policy.view(not_possible.shape[0], -1).softmax(-1)
        # topk_policy_values, topk_policy_indices = torch.topk(policy, 10, -1)
        # selected_policy = torch.multinomial(topk_policy_values, num_samples=top_k)
        # policies = torch.gather(topk_policy_indices, 1, selected_policy)
        _, policies = torch.topk(policy, top_k, -1)
        predicted_pos = [self.format_pos(batch, ncol=board_state.shape[-1]) for batch in policies.tolist()]
        return predicted_pos[0], value.squeeze(0)

    def forward(self, state, top_k=3, **kwargs):
        return self.predict_next_pos(state, top_k=top_k)

class Node():
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        # self.parent = parent
        self.childrens = []
    
    def set(self, x): self.value = x
    
    def add_node(self, node):
        self.childrens.append(node)

    def __repr__(self, depth):
        if self.childrens.__len__() == 0:
            return f"{self.name}-{self.value:.3f}"
        return {self.name: [children.__repr__(depth-1) for children in self.childrens]}
    
    def __str__(self):
        return f"<Node name={self.name} value={self.value} childrens={self.childrens}>"

    def _get(self, data_dict, graph, parent_id=None, last=False):
        if last:
            my_id = str(uuid.uuid4())
            graph.node(my_id, str(data_dict))
            graph.edge(my_id, parent_id)
            return graph

        for data in data_dict.keys():
            my_id = str(uuid.uuid4())
            graph.node(my_id, data)

            if parent_id != None:
                graph.edge(my_id, parent_id)

            for child in data_dict[data]:
                self._get(data_dict=child, graph=graph, parent_id=my_id, last=type(child)==type("a"))
        
        return graph

    def viz(self, depth):
        viz_data = self.__repr__(depth)
        graph = graphviz.Digraph()
        
        graph = self._get(viz_data, graph, None)

        return graph

class MinimaxWithAB(PytorchAgent):
    MIN = "min"
    MAX = "max"

    def __init__(self, max_search_node, **kwargs):
        super().__init__(**kwargs)
        self.max_search_node = max_search_node

    def whose_turn(self, board_state):
        return int(board_state[0].sum() == board_state[1].sum())

    # Search 3 highest value vertex until reaching the maximum depth 
    def minimax_search(self, node: Node, my_turn, strategy, board_state, depth, alpha, beta):
        if depth == 0:
            _, value = self.model_predict(state=board_state)

            if strategy == MinimaxWithAB.MAX:
                value = 1 - value

            node = node.add_node(Node("Bottom", value.item()))

            return value, _

        next_turn = 1 - my_turn
        selected_poses, _ = self.predict_next_pos(board_state=board_state, top_k=self.max_search_node)

        cur_value = float("inf") if strategy == MinimaxWithAB.MIN else -float("inf")
        origin = None

        for i, next_pos in enumerate(selected_poses):
            cur_node = Node(f"{next_pos}({depth}/{i})")
            col, row = next_pos
            new_board_state = board_state.copy()
            print(f"--- DEPTH {depth}  ---")
            new_board_state[my_turn, row, col] = 1

            strategy = MinimaxWithAB.MIN if strategy == MinimaxWithAB.MAX else MinimaxWithAB.MAX
            value, origin = self.minimax_search(node=cur_node, my_turn=next_turn, strategy=strategy, board_state=new_board_state, depth=depth-1, alpha=alpha, beta=beta)

            cur_node.set(value.item())
            node.add_node(cur_node)

            if strategy == MinimaxWithAB.MIN:
                cur_value = min(cur_value, value)
                origin = next_pos
                beta = min(beta, cur_value)
                if value <= alpha:
                    break
            elif strategy == MinimaxWithAB.MAX:
                cur_value = max(cur_value, value)
                origin = next_pos
                alpha = max(alpha, cur_value)
                if value >= beta:
                    break

        # if depth == 3:
            # graph = node.viz(depth=depth)
            # graph.view()

        return value, origin

    def forward(self, board_state, turn, max_depth=3):
        node = Node("Root")
        # Agent wants to maximize the value he might receive.
        value, next_pos = self.minimax_search(node=node, my_turn=turn, strategy=MinimaxWithAB.MAX, board_state=board_state, depth=max_depth, alpha=-float("inf"), beta=float("inf"))
        
        if DEBUG >= 2: print(f"BEST Searching Result --- {value}")

        return [next_pos], value