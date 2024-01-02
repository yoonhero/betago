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
import torch.multiprocessing as mp
from einops import rearrange

from gomu.gomuku import board
from gomu.base import PolicyValueNet, get_total_parameters
from gomu.bot import *
from gomu.game_tree import Graph
from gomu.helpers import DEBUG

from train_base import get_loss, GOMUDataset, training_one_epoch, save_result
from elo import ELO
from shared_adam import SharedAdam
from data_utils import TempDataset

simulation_no = int(os.getenv("MAX", 100))
max_vertex = int(os.getenv("MAX_VERTEX", 5))
save_term = int(os.getenv("SAVE_TERM", 1))
log = bool(int(os.getenv("LOG", 0)))
eval_term = int(os.getenv("EVAL_TERM", 1))
is_mp = bool(int(os.getenv("MP", 0)))
device = os.getenv("DEVICE", "cpu")
UPDATE_GLOBAL_ITER = 2
TOTAL_ELO_SIM = 50

save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
Path("./tmp").mkdir(exist_ok=True)
save_base_path.mkdir(exist_ok=True)
(save_base_path / "ckpt").mkdir(exist_ok=True)
(save_base_path / "evalresults").mkdir(exist_ok=True)
(save_base_path / "trainresults").mkdir(exist_ok=True)

max_turn = 50
model_elo = 100
base_elo = 500

nrow = 20
ncol = 20
n_to_win = 5
game_info = GameInfo(nrow=nrow, ncol=ncol, n_to_win=n_to_win)

zero_state = np.zeros((2, nrow, ncol))

channels = [2, 64, 128, 256, 128, 64, 32, 1]
dropout = 0.5
# model = PolicyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=dropout)
# model.to(device)
model = load_base(game_info, first_channel=2, device=device, cpk_path="./models/1224-256.pkl")
model.share_memory()

agent = PytorchAgent(model=model, device=device, n_to_win=n_to_win, with_history=False)

# Training Hyperparameters
batch_size = 256
learning_rate = 5e-4
weight_decay = 0.1
    
optimizer_ckp = torch.load("./models/1224-256.pkl", map_location=torch.device(device))["optim"]
optimizer = SharedAdam(model.parameters(), lr=learning_rate, betas=(0.92, 0.999))
optimizer.load_state_dict(optimizer_ckp)
total_parameters = get_total_parameters(model)
print(total_parameters)

class GGraph(Graph):
    def init(self):
        self._data = {}
        self.graph = defaultdict(list)
    def root(self, root_node): self._data[root_node.id] = root_node
    @property
    def data(self): return self._data

class MCTSNode():
    def __init__(self, state, turn, mcts_graph, action=None, parent=None, updated=None):
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
        self.mcts_graph = mcts_graph

        self.turn = turn
        if updated == None:
            self.updated = []
        else:
            self.updated = updated
    
    def init(self):
        self.childrens = []
        self._results = defaultdict(int)
        self._untried_actions = self.untried_actions()
        self.updated = []

    def untried_actions(self):
        _untried_actions = self.get_legal_actions()
        return _untried_actions

    def maximum_top_k(self, state):
        empty_spaces = (state.sum(0)!=1).sum()
        top_k = max_vertex if empty_spaces > max_vertex else empty_spaces
        return top_k

    def get_legal_actions(self):
        top_k = self.maximum_top_k(self.state)
        if top_k != 0:
            next_poses, _ = agent.predict_next_pos(board_state=self.state, top_k=top_k)
            return next_poses
        return []

    def q(self):
        win = self._results[1]
        lose = self._results[-1]
        return win - lose

    def n(self):
        return self.visit_count
    
    def expand(self):
        action = self._untried_actions.pop()
        next_state = agent.get_new_board_state(board_state=self.state, next_pos=action, my_turn=self.turn)
        child_node = MCTSNode(next_state, turn=1-self.turn, parent=self, action=action, updated=self.updated, mcts_graph=self.mcts_graph)
        self.childrens.append(child_node)

        self.mcts_graph.addEdge(self, child_node)

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
            top_k = self.maximum_top_k(current_rollout_state)
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
        self.updated.append(self.id)

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

def get_train_data(updated, mcts_graph):
    train_data = []
    real_updated = set(updated)
    # (State, Next Action Pi Distribution, expected value=.q)
    for node_key in real_updated:
        cur_node = mcts_graph.data[node_key]
        tensor_state = torch.tensor(cur_node.state)
        turn = cur_node.turn
        if cur_node.childrens.__len__() == 0:
            continue
        next_actions = [(children.action, 1-children.q()/children.n()) for children in cur_node.childrens]
        # next_actions = [(children.action, children.n()) for children in cur_node.childrens]
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
    print(f"Total Train Data Size: {len(train_data)}")
    return train_data

def push_and_pull(train_data, opt, lnet, gnet, res_queue, g_elo, total_step, scenario_turn, name):
    print(f"Update {name}")
    train_loader = torch.utils.data.DataLoader(TempDataset(train_data, device=device), batch_size=batch_size, shuffle=True)
    
    lnet.train()
    gnet.train()
    opt.zero_grad()

    def shared_training_one_epoch(loader, net, epoch, nrow, ncol, save_base_path):
        _loss = []
        num_correct = 0
        num_samples = 0

        with tqdm.tqdm(loader) as pbar:
            for step, (X, Y, win) in enumerate(pbar):
                net.train()
                policy, value = net(X)
                loss = get_loss(policy, value, Y, win, nrow, ncol)

                loss.backward()
                _loss.append(loss.item()) 

                pbar.set_description(f"{epoch}/{step}")
                pbar.set_postfix(loss=_loss[-1])
                
                num_correct += ((value>0.5)==(win!=0)).sum() / win.sum() * value.size(0)
                num_samples += value.size(0)
            
            save_result(X[0], Y[0], policy[0], save_base_path, nrow=nrow, ncol=ncol, epoch=epoch)
        
        return sum(_loss) / len(_loss), num_correct / num_samples

    training_loss, training_acc = shared_training_one_epoch(train_loader, lnet, total_step, nrow=nrow, ncol=ncol, save_base_path=save_base_path)

    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    lnet.load_state_dict(gnet.state_dict())

    # if (scenario_turn+1) % eval_term == 0:
    g_elo.value = ELO(model_elo=g_elo.value, base_elo=base_elo, model=gnet, total_play=TOTAL_ELO_SIM, game_info=game_info, device=device)

    res_queue.put((training_loss, training_acc, g_elo.value, total_step, scenario_turn, name))
    return

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_elo, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_elo, self.res_queue = global_elo, res_queue
        self.gnet, self.opt = gnet, opt

        self.lnet = load_base(game_info=game_info, first_channel=2, device=device, cpk_path="./models/1224-256.pkl")
        self.updated = []
        self.mcts_graph = GGraph()
        self.root_node = MCTSNode(state=zero_state, turn=0, parent=None, mcts_graph=self.mcts_graph)
        self.mcts_graph.root(root_node=self.root_node)

    def run(self):
        total_step = 1
        while True:
            self.mcts_graph.init()
            self.root_node.init()
            self.mcts_graph.root(root_node=self.root_node)
            
            for scenario_turn in range(max_turn):
                print(f"Simulating {self.name}")
                self.root_node.simulate()

                updated = self.root_node.updated
                train_data = get_train_data(updated, mcts_graph=self.mcts_graph)

                # update global and assign to local net
                push_and_pull(train_data, self.opt, self.lnet, self.gnet, self.res_queue, self.g_elo, total_step, scenario_turn, self.name)

            total_step += 1

        self.res_queue.put(None)

def main(logger):
    global_elo, res_queue = mp.Value('d', 100.), mp.Queue()
    # total_workers = mp.cpu_count()-1
    total_workers = 12
    workers = [Worker(gnet=model, opt=optimizer, global_elo=global_elo, name=i, res_queue=res_queue) for i in range(total_workers)]
    print(f"Total {len(workers)} workers.")
    [w.start() for w in workers]
    while True:
        r = res_queue.get()

        if r == None:
            break    
        
        train_loss, train_accuracy, current_elo, total_step, scenario_turn, name  = r

        if (scenario_turn+1) % save_term == 0:
            save_path = save_base_path / f"ckpt/epoch-{total_step}-{scenario_turn}-{name}.pkl"
            print(f"Save in {save_path}")
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, save_path)

        if log:    
            logger.log({"train/loss": train_loss, "train/acc": train_accuracy, "elo": current_elo})
    [w.join() for w in workers]

def normal_train(logger):
    global model_elo, base_elo
    epoch = 0

    mcts_graph = GGraph()
    root = MCTSNode(state=zero_state, turn=0, parent=None, mcts_graph=mcts_graph)
    mcts_graph.root(root_node=root)

    while True:
        epoch += 1
        mcts_graph.init()
        root.init()
        mcts_graph.root(root_node=root)

        for scenario_turn in range(max_turn):
            print("Simulating....")
            root.simulate()

            if DEBUG >= 3:
                mcts_graph.viz(root).view()

            train_data = get_train_data(updated=root.updated, mcts_graph=mcts_graph)
            
            train_loader = torch.utils.data.DataLoader(TempDataset(train_data, device=device), batch_size, shuffle=True)

            model.train()
            train_loss, train_accuracy = training_one_epoch(train_loader, model, optimizer, True, epoch, nrow=nrow, ncol=ncol, save_base_path=save_base_path)
            # test_loss, test_accuracy = training_one_epoch(test_loader, model, optimizer, False, epoch, nrow=nrow, ncol=ncol)

            if (scenario_turn+1) % eval_term == 0:
                model_elo = ELO(model_elo=model_elo, base_elo=base_elo, model=model, total_play=TOTAL_ELO_SIM, game_info=game_info, device=device)

            if log:
                logger.log({"train/loss": train_loss, "train/acc": train_accuracy, "elo": model_elo})

            if (scenario_turn+1) % save_term == 0:
                save_path = save_base_path / f"ckpt/epoch-{epoch}-{scenario_turn}.pkl"
                torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, save_path)
            
            print(f"{epoch}/{scenario_turn}th EPOCH DONE!! ---- Train: {train_loss}")



if __name__ == "__main__":
    mp.set_start_method('spawn')

    if log:
        logger = wandb.init(
            project="AlphaGomu"
        )

    if is_mp:
        main(logger)
    else:
        normal_train(logger)