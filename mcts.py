#! /usr/bin/env python3
# MCTS Alpha-Zero Implementatino on GO.
import uuid
from queue import Queue
import os
from collections import defaultdict
import math
import random
import wandb
import time
from pathlib import Path

import torch
import numpy as np
import torch.optim as optim
from einops import rearrange

from gomu.gomuku import board
from gomu.base import PolicyValueNet, get_total_parameters
from gomu.bot import *
from gomu.game_tree import Graph
from gomu.helpers import DEBUG

from train_base import get_loss, GOMUDataset, training_one_epoch

simulation_no = int(os.getenv("MAX", 100))
max_vertex = int(os.getenv("MAX_VERTEX", 5))
save_term = int(os.getenv("SAVE_TERM", 1))
log = bool(int(os.getenv("LOG", 0)))

class GGraph(Graph):
    def init(self):
        self._data = {}
        self.graph = defaultdict(list)
    def root(self, root_node): self._data[root_node.id] = root_node
    @property
    def data(self): return self._data

class MCTSNode():
    def __init__(self, state, turn, action=None, parent=None):
        self.state = state
        self.parent = parent
        self.childrens = []
        self.action = action

        self.visit_count = 0

        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0

        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.id = str(uuid.uuid4())

        self.turn = turn
    
    def init(self):
        self.childrens = []
        self._results = defaultdict(int)
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        _untried_actions = self.get_legal_actions()
        return _untried_actions

    def get_legal_actions(self):
        next_poses, _ = agent.predict_next_pos(board_state=self.state, top_k=max_vertex)
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
        child_node = MCTSNode(next_state, turn=1-self.turn, parent=self, action=action)
        self.childrens.append(child_node)

        mcts_graph.addEdge(self, child_node)

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
        model.eval()

        current_rollout_state = self.state
        turn = self.turn

        depth = 0

        while not GoMuKuBoard.is_game_done(board_state=current_rollout_state, turn=turn, n_to_win=n_to_win):
            depth += 1
            turn = 1 - turn
            empty_spaces = (current_rollout_state.sum(0)!=1).sum()
            top_k = max_vertex if empty_spaces > max_vertex else empty_spaces
            if top_k == 0:
                if DEBUG >= 2:
                    print(current_rollout_state)
                return 1/2
            possible_moves, _ = agent.predict_next_pos(board_state=current_rollout_state, top_k=top_k)

            action = self.rollout_policy(possible_moves)
            current_rollout_state = agent.get_new_board_state(board_state=current_rollout_state, next_pos=action, my_turn=turn)
        
        if DEBUG >= 3:
            print(f"Current Depth: {depth}")

        if DEBUG >= 3:
            GoMuKuBoard.viz(current_rollout_state).show()

        if self.turn == turn:
            return 1
        return -1

    # def rollout(self)
    def backpropagate(self, result):
        self.visit_count += 1
        if result != 0.5:
            self._results[result] += 1
        else:
            self._results[1] += 0.5
            self._results[-1] += 0.5

        # Add the node for training nn.
        updated.append(self.id)

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return not int(self._untried_actions.__len__())

    def score(self, c=0.1):
        ucb_score = self.q() / (self.n()+1) + c * math.sqrt(2 * math.log(self.parent.n()) / (self.n()+2))
        return ucb_score
    
    def best_child(self, c_param=0.1):
        choices_weight = [children.score(c=c_param) for children in self.childrens]
        return self.childrens[np.argmax(choices_weight)]
    
    def _tree_policy(self):
        current_node = self

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node
    
    def simulate(self):
        for i in tqdm.trange(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return

    def __repr__(self) -> str:
        if self.parent != None:
            return f"MCTS(value={self.score()}, pos={self.action})"
        
        return f"MCTS(root)"


nrow = 10
ncol = 10
n_to_win = 5
channels = [2, 64, 128, 256, 128, 64, 32, 1]
dropout = 0.5

device = "mps"

model = PolicyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=dropout)
model.to(device)

agent = PytorchAgent(model=model, device=device, n_to_win=n_to_win, with_history=False)

updated = []
mcts_graph = GGraph()

zero_state = np.zeros((2, nrow, ncol))
root = MCTSNode(state=zero_state, turn=0, parent=None)
mcts_graph.root(root_node=root)

# Training Hypterparameters
batch_size = 12
learning_rate = 0.001
weight_decay = 0.1
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_parameters = get_total_parameters(model)
print(total_parameters)

save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
save_base_path.mkdir(exist_ok=True)
(save_base_path / "ckpt").mkdir(exist_ok=True)
(save_base_path / "evalresults").mkdir(exist_ok=True)
(save_base_path / "trainresults").mkdir(exist_ok=True)

# test_dataset = GOMUDataset(50, device=device)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

epoch = 0
max_turn = 50

if log:
    run = wandb.init(
        project="AlphaGomu"
    )

class TempDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.total = len(data)
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

        state = state.to(torch.float32).to(device)
        pi = pi.to(torch.float32).to(device)
        value = value.to(torch.float32).to(device)
        return state, pi, value
    def __len__(self):
        return self.total

while True:
    epoch += 1
    mcts_graph.init()
    root.init()
    mcts_graph.root(root_node=root)

    for turn in range(max_turn):
        # initialize the udpated
        updated = []

        print("Simulating....")
        root.simulate()

        if DEBUG >= 3:
            mcts_graph.viz(root).view()

        train_data = []
        real_updated = set(updated)
        print(f"Total Train Data: {len(real_updated)}")
        # (State, Next Action Pi Distribution, expected value=.q)
        for node_key in real_updated:
            cur_node = mcts_graph.data[node_key]
            tensor_state = torch.tensor(cur_node.state)
            turn = cur_node.turn
            if cur_node.childrens.__len__() == 0:
                continue
            # next_actions = [(children.next_pos, 1-children.q()/children.n()) for children in cur_node.childrens]
            next_actions = [(children.action, children.n()) for children in cur_node.childrens]
            pi = torch.zeros((1, nrow, ncol))
            # Is V=r+gammaQ?
            for next_action_data in next_actions:
                next_action, expected_value = next_action_data
                col, row = next_action
                pi[0, row, col] = expected_value
            # Q normalization
            pi = pi / pi.sum()
            expected_value = torch.tensor([cur_node.q()/cur_node.n()])
            item = (tensor_state, pi, expected_value)
            train_data.append(item)
        
        train_loader = torch.utils.data.DataLoader(TempDataset(train_data), batch_size, shuffle=True)

        model.train()
        train_loss, train_accuracy = training_one_epoch(train_loader, model, optimizer, True, epoch, nrow=nrow, ncol=ncol, save_base_path=save_base_path)
        # test_loss, test_accuracy = training_one_epoch(test_loader, model, optimizer, False, epoch, nrow=nrow, ncol=ncol)

        if log:
            run.log({"train/loss": train_loss, "train/acc": train_accuracy})

        if (epoch+1) % save_term == 0:
            save_path = save_base_path / f"ckpt/epoch-{epoch}.pkl"
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, save_path)

        print(f"{epoch}/{turn}th EPOCH DONE!! ---- Train: {train_loss}")

