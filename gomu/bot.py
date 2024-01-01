from typing import Any
import numpy as np
import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from einops import rearrange

from .helpers import DEBUG
from .gomuku.board import GoMuKuBoard
from .game_tree import Node, Graph
from .gomuku.errors import PosError
from .viz import tensor2gomuboard

# Base Agent Structure
class Agent():
    def __init__(self, n_to_win, with_history, **kwargs):
        self.n_to_win = n_to_win
        self.turn = None
        self.with_history = with_history
        self.history = []

    def set_turn(self, turn):
        self.turn = turn

    def forward(self, board_state, **kwargs):
        return NotImplemented

    def __call__(self, board_state, **kwds: Any) -> Any:
        next_pos, value = self.forward(board_state, **kwds)
        if self.validate(board_state=board_state, next_pos=next_pos):
            return next_pos, value

    # Validate the predicted zone was possible or not.
    def validate(self, board_state, next_pos):
        _, n_col, n_row = board_state.shape
        col, row = next_pos

        # Bounding Check
        if n_col <= col or n_row <= row:
            raise PosError(col, row)

        # Check Is Empty Space
        if (board_state[:, row, col]==1).any():
            raise PosError(col, row)

        return True

    # Give a heuristic value to prevent ugly movement.
    def heuristic(self, board_state, my_turn):
        heuristic_value = 0

        is_game_done = GoMuKuBoard.is_game_done(board_state, my_turn, self.n_to_win)
        if is_game_done:
            heuristic_value = float("inf")
        
        return heuristic_value

    def get_not_free_space(self, board_state):
        not_free_space = board_state.sum(0)
        return not_free_space
    
    def format_pos(self, indices, ncol):
        return [(idx%ncol, idx//ncol) for idx in indices]

    def update_history(self, selected_pos):
        if selected_pos not in self.history and selected_pos != None:
            self.history.append(selected_pos)


# Actually just stupid guy
class RandomMover(Agent):
    def forward(self, board_state, **kwargs):
        not_free_space = self.get_not_free_space(board_state=board_state)
        not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32)
        policy = torch.ones_like(not_possible)
        not_possible[not_possible!=0] = -float("inf")
        policy += not_possible
        policy = policy.view(-1).softmax(0)
        next_pos_indices = torch.multinomial(policy, num_samples=1).tolist()
        next_pos = self.format_pos(next_pos_indices, not_possible.shape[-1])
        return next_pos[0], 0

# Simple Pytorch Agent
class PytorchAgent(Agent):
    def __init__(self, model, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device

    # Always first element is me~
    def preprocess_state(self, state):
        torch_state = torch.from_numpy(state).to(dtype=torch.float32, device=self.device)

        total_BLACK = torch_state[0].sum()
        total_WHITE = torch_state[1].sum()
        if total_BLACK != total_WHITE:
            torch_state = torch.roll(torch_state, 1, 0)

        return torch_state

    def make_history_state(self, history, nrow, ncol, history_size=2):
        if not isinstance(history[0], list):
            prev_moves = history[-history_size:]
            if prev_moves.__len__() < 2:
                prev_moves = [None] * (2-prev_moves.__len__()) + prev_moves 
            zero_board_state = np.zeros([history_size, nrow, ncol])
            for i, prev_move in enumerate(prev_moves):
                if prev_move != None:
                    col, row = prev_move
                    zero_board_state[i, row, col] = 1
            
            return torch.from_numpy(zero_board_state).to(torch.float32).to(self.device)
        else:
            result = []
            for hi in history:
                tensor_hi = self.make_history_state(history=hi, nrow=nrow, ncol=ncol, history_size=history_size)
                result.append(tensor_hi)
            return torch.cat(result, dim=0).to(torch.float32).to(self.device)

    # State: Board State.
    # History: Full History list of position.
    @torch.no_grad()
    def model_predict(self, state, history=None):
        _, ncol, nrow = state.shape

        if isinstance(state, np.ndarray):
            state = self.preprocess_state(state)
        
        if self.with_history:
            if history == None:
                history = self.history
            torch_history = self.make_history_state(history, nrow, ncol)
            state = torch.cat([torch_history, state]).unsqueeze(0)
        else:
            state = state.unsqueeze(0)

        policy, value = self.model(state)
        # policy[:, :, 0], policy[:, :, 1] = policy[:, :, 1], policy[:, :,0]
        return policy, value
   
    def get_new_board_state(self, board_state, next_pos, my_turn):
        if self.validate(board_state, next_pos=next_pos):
            col, row = next_pos
            new_board_state = board_state.copy()
            new_board_state[my_turn, row, col] = 1
            return new_board_state
    
    def predict_next_pos(self, board_state, top_k, temperature=1, best=False, history=None):
        # assert top_k >= 3, "Please top_k is greater than 3."
        policy, value = self.model_predict(board_state, history=history)

        not_free_space = self.get_not_free_space(board_state=board_state)
        not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        B, ncol, nrow = not_possible.shape
        # not_possible[not_possible!=0] = -float("inf")
        not_possible = 1 - not_possible
        # policy += not_possible

        if DEBUG >= 3:
            img = tensor2gomuboard(policy[0], nrow, ncol, softmax=True, scale=10)
            img.show()

        policy = policy.view(B, -1).softmax(-1)
        policy = policy * not_possible.view(B, -1)
            
        # indices = torch.arange(policy.shape[-1]).repeat(B, 1)
        if best:
            sampling_k = top_k
        else:
            sampling_k = top_k + 1
        topk_policy_values, topk_policy_indices = torch.topk(policy, sampling_k, -1)
        selected_policy = torch.multinomial(topk_policy_values/temperature, num_samples=top_k)
        policies = torch.gather(topk_policy_indices, 1, selected_policy)
        # _, policies = torch.topk(policy, top_k, -1)
        predicted_pos = [self.format_pos(batch, ncol=board_state.shape[-1]) for batch in policies.tolist()]
        # if not batch:
        return predicted_pos[0], value.squeeze(0)
        # else:
            # return predicted_pos, value

    def forward(self, board_state, top_k=1, **kwargs):
        next_poses, value = self.predict_next_pos(board_state, top_k=top_k)
        return next_poses[0], value.cpu().item()


class MinimaxWithAB(PytorchAgent):
    MIN = "min"
    MAX = "max"

    def __init__(self, max_search_vertex, max_depth, **kwargs):
        super().__init__(**kwargs)
        self.max_search_vertex = max_search_vertex
        self.max_depth = max_depth
        self.gamma = 1e-1

    def whose_turn(self, board_state):
        return int(board_state[0].sum() == board_state[1].sum())

    # Search n highest value vertex with alpha-beta prunning until reaching the maximum depth. 
    def minimax_search(self, parent_node: Node, game_tree: Graph, my_turn, strategy, board_state, depth, alpha, beta, history=None, root=False):
        heuristic_value = self.heuristic(board_state=board_state, my_turn=my_turn)
        if strategy == MinimaxWithAB.MAX: heuristic_value *= -1
        if heuristic_value != 0: return heuristic_value, None

        if depth == 0:
            if DEBUG >= 3:
                GoMuKuBoard.viz(board_state).show()
            _, tensor_value = self.model_predict(state=board_state)
            value = tensor_value.cpu().item()

            # Depending on the role, redefine the value for minimax searching.
            # Maximum Player is BOT. They want to maximize their winning probability.
            # Doing so, on the last depth, Stragety==MAX
            if strategy == MinimaxWithAB.MAX:
                value = 1 - value

            parent_node.set(value)

            return value, None

        if history == None:
            history = self.history
        next_turn = 1 - my_turn
        selected_poses, current_state_value = self.predict_next_pos(board_state=board_state, top_k=self.max_search_vertex, history=history, temperature=0.1)
        # current_state_value = current_state_value.cpu().item()
        # if strategy == MinimaxWithAB.MIN: current_state_value = 1 - current_state_value

        cur_value = float("inf") if strategy == MinimaxWithAB.MIN else -float("inf")
        origin = None
        new_strategy = MinimaxWithAB.MIN if strategy == MinimaxWithAB.MAX else MinimaxWithAB.MAX

        loader = tqdm.tqdm(selected_poses) if root else selected_poses

        for next_pos in loader:
            cur_node = Node(f"{next_pos}")
            new_board_state = None
            _history = deepcopy(history)
            _history.append(next_pos)

            try:
                new_board_state = self.get_new_board_state(board_state, next_pos, my_turn)
            except PosError:
                if DEBUG >= 1:
                    print(f"There was error during getting the new board state. --- Pos {next_pos}")
                continue
            
            value, _ = self.minimax_search(parent_node=cur_node, game_tree=game_tree, my_turn=next_turn, strategy=new_strategy, board_state=new_board_state, history=_history, depth=depth-1, alpha=alpha, beta=beta)
            if DEBUG >= 3:
                print(f"--- DEPTH {depth}  ---")
                print(f"Action Value: {value} | Strategy: {strategy}")
            
            # recursive_value = current_state_value + self.gamma * value
            recursive_value = value

            if strategy == MinimaxWithAB.MIN:
                cur_value = min(cur_value, recursive_value)
                if cur_value == recursive_value:
                    origin = next_pos
                beta = min(beta, cur_value)
                if value <= alpha:
                    break
            elif strategy == MinimaxWithAB.MAX:
                cur_value = max(cur_value, recursive_value)
                if cur_value == recursive_value:
                    origin = next_pos
                alpha = max(alpha, cur_value)
                if value >= beta:
                    break

            # Making a connection in Game Tree 
            cur_node.set(recursive_value)
            game_tree.addEdge(parent_node, cur_node)

        parent_node.set(cur_value)
        if root and DEBUG >= 2:
            digraph = game_tree.viz(parent_node)
            digraph.view()

        return cur_value, origin

    def forward(self, board_state, turn, **kwargs):
        node = Node("Root")
        game_tree = Graph()
        # Agent wants to maximize the value he might receive.
        value, next_pos = self.minimax_search(game_tree=game_tree, parent_node=node, my_turn=turn, strategy=MinimaxWithAB.MAX, board_state=board_state, depth=self.max_depth, alpha=-float("inf"), beta=float("inf"), root=True)
        
        if DEBUG >= 1: print(f"BEST Searching Result --- {value}")

        return next_pos, value