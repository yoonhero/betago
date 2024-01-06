#! /usr/bin/env python3
# MCTS Alpha-Zero Implementatino on GO.
import uuid
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
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
from gomu.base import PolicyValueNet, NewPolicyValueNet, get_total_parameters
from gomu.bot import *
from gomu.game_tree import Graph
from gomu.helpers import DEBUG

from train_base import get_loss, training_one_epoch, save_result
from elo import ELO
from shared_adam import SharedAdam
from gomu.data_utils import TempDataset

simulation_no = int(os.getenv("MAX", 100))
max_vertex = int(os.getenv("MAX_VERTEX", 5))
save_term = int(os.getenv("SAVE_TERM", 1))
log = bool(int(os.getenv("LOG", 0)))
eval_term = int(os.getenv("EVAL_TERM", 1))
is_mp = bool(int(os.getenv("MP", 0)))
device = os.getenv("DEVICE", "cpu")
ckp = os.getenv("CKP", "./models/1224-256.pkl")
UPDATE_GLOBAL_ITER = 2
TOTAL_ELO_SIM = 50

max_turn = 2
model_elo = 100
base_elo = 500

games = {"ttt": GameInfo(nrow=3, ncol=3, n_to_win=3), "gomu": GameInfo(20, 20, 5)}
game_info = games["gomu"]
nrow, ncol, n_to_win = game_info()

zero_state = np.zeros((2, nrow, ncol))

# Training Hyperparameters
batch_size = 256
learning_rate = 5e-4
weight_decay = 0.1

def normal_dist(x, y, xmu, ymu, sig):
    return 1/np.sqrt(2*np.pi*pow(sig, 2)) * np.exp((-pow((x-xmu),2)-pow((y-ymu), 2))/(2*pow(sig, 2)))

def make_circular_heatmap(ncol, nrow):
    xmu = (ncol-1) / 2
    ymu = (nrow-1) / 2
    sig = (xmu+ymu)/4

    heatmap = np.zeros((nrow, ncol))
    
    for row in range(nrow):
        for col in range(ncol):
            val = normal_dist(x=col, y=row, xmu=xmu, ymu=ymu, sig=sig)
            heatmap[row, col] = val
    
    return heatmap

class EGreedyAgent(PytorchAgent):
    @staticmethod
    def eps(n):
        return 1 / (1+np.exp(-0.2*n))
    
    def get_policy_and_value(self, board_state):
        policy, value = self.model_predict(board_state)

        policy = torch.softmax(policy, 1).squeeze(0).cpu()
        not_free_space = self.get_not_free_space(board_state=board_state)
        not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32).unsqueeze(0)
        policy *= (1-not_possible)
        policy /= policy.sum()

        value = value.item()

        return policy, value

    def predict_egreedy_pos(self, board_state, top_k, temperature=1, eps=0.2):
        num_best = math.ceil(top_k * (1-eps))
        num_explore = top_k - num_best
        best_action, _ = self.predict_next_pos(board_state, num_best, temperature, best=True, history=False)

        if num_explore > 0:
            circular_zone = make_circular_heatmap(ncol=ncol, nrow=nrow)
            circular_zone = torch.from_numpy(circular_zone).to(dtype=torch.float32, device=self.device)
            not_free_space = self.get_not_free_space(board_state=board_state)
            not_possible = torch.from_numpy(not_free_space).to(dtype=torch.float32, device=self.device).unsqueeze(0)
            possible = (1 - not_possible)*circular_zone
            possible = possible.view(1, -1)
            selected_policy = torch.multinomial(possible/temperature, num_samples=num_explore)
            random_action = [self.format_pos(batch, ncol=board_state.shape[-1]) for batch in selected_policy.tolist()][0]
        else: random_action = []

        action = best_action + random_action
        return list(set(action))

class GGraph(Graph):
    def init(self):
        self._data = {}
        self.graph = defaultdict(list)
    def root(self, root_node): self._data[root_node.id] = root_node
    @property
    def data(self): return self._data

class MCTSNode():
    def __init__(self, state, mcts_graph, args, action=None, prior=0, parent=None, board_env=None):
        self.state = state
        self.parent = parent
        self.board_env: GoMuKuBoard = board_env
        self.childrens = []
        self.action = action
        self.prior = 0
        self.visit_count = 0
        self.value_sum = 0

        # self.agent: EGreedyAgent = agent
        self.id = str(uuid.uuid4())
        self.mcts_graph = mcts_graph

        self.args = args
    
    def expand(self, policy: torch.Tensor):
        for i, prob in enumerate(policy.view(-1).tolist()):
            action = self.board_env.format_pos(i)
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.agent.get_new_board_state(board_state=child_state, next_pos=action, my_turn=self.turn)
                child_state = GoMuKuBoard.change_perspective(child_state)
                
                child_node = MCTSNode(child_state, parent=self, action=action, updated=self.updated, mcts_graph=self.mcts_graph, agent=self.agent, board_env=self.board_env)
                self.childrens.append(child_node)

                self.mcts_graph.addEdge(self, child_node)

        return child_node

    def is_terminal_node(self):
        return self.game_result() != 0

    def is_fully_expanded(self):
        return not int(self._untried_actions.__len__())

    def score(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        # ucb_score = self.q() / (self.n()+1) + c * math.sqrt(2 * math.log(self.parent.n()) / (self.n()+2))
        return q_value + self.args["C"] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def best_child(self):
        choices_weight = [self.score(children) for children in self.childrens]
        return self.childrens[np.argmax(choices_weight)]
    
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        if self.parent:
            self.parent.backpropagate(-value)

    def __repr__(self) -> str:
        if self.parent:
            return f"MCTSNode(pos={self.action})"
    
        return f"MCTSNode(root)"
    
class MCTS(Graph):
    def init(self, agent, args):
        self._data = {}
        self.graph = defaultdict(list)
        self.agent: EGreedyAgent = agent
        self.board_env = GoMuKuBoard(nrow, ncol, n_to_win)
        self.args = args

    def set_root(self, root_node): self._data[root_node.id] = root_node
    
    @torch.no_grad()
    def search(self, state):
        root = MCTSNode(state=state, parent=None, mcts_graph=self.mcts_graph, args=self.args, board_env=self.board_env)

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.best_child()
            
            value, is_terminal = self.board_env.get_value_and_terminated(node.state)

            if not is_terminal:
                policy, value = self.agent.get_policy_and_value(node.state)
                
                node.expand(policy)
            
            node.backpropagate(value)

        action_probs = np.zeros((1, 20, 20))
        for child in root.childrens:
            col, row = child.action
            action_probs[0, row, col] = child.visit_count
        action_probs /= np.sum(action_probs)

    
        if DEBUG >= 3:
            self.viz(root).view()

        return action_probs

class Zero():
    def __init__(self, model, optimizer, agent, device, args, save_base_path, logger):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.agent: EGreedyAgent = agent
        self.mcts = MCTS(agent, args)
        self.device = device
        self.save_base_path = save_base_path
        self.logger = logger

    def self_play(self):
        memory = []
        state = zero_state.copy()
        player = 0

        while True:
            initial_state = GoMuKuBoard.change_perspective(state)
            action_probs = self.mcts.search(initial_state)

            memory.append((initial_state, action_probs, player))
            
            action = torch.multinomial(action_probs.view(-1), num_samples=1)
            action = self.board_env.format_pos(action)

            state = self.agent.get_new_board_state(board_state=initial_state, next_pos=action, my_turn=player)

            value, is_terminal = self.board_env.get_value_and_terminated(state)

            if is_terminal:
                return_memory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    return_memory.append((hist_state, hist_action_probs, hist_outcome))
                return return_memory

            player = 1 - player
    
    def train(self, data, epoch):
        train_loader = torch.utils.data.DataLoader(TempDataset(data, device=self.device), batch_size, shuffle=True)

        train_loss, train_accuracy = training_one_epoch(train_loader, self.model, self.optimizer, True, epoch, nrow=nrow, ncol=ncol, save_base_path=save_base_path)

        return train_loss, train_accuracy

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for self_play_iteration in tqdm.trange(self.args["num_self_play_iterations"]):
                memory += self.self_play()
            
            self.model.train()
            for epoch in tqdm.trange(self.args["num_epochs"]):
                train_loss, train_accuracy = self.train(memory, epoch=self.args["num_epochs"] * iteration + epoch)

                if (epoch+1) % eval_term == 0 and n_to_win == 5:
                    model_elo = ELO(challenger_elo=model_elo, critic_elo=base_elo, challenger=self.model, total_play=TOTAL_ELO_SIM, game_info=game_info, device=device, op_ckp=ckp)

                if log:
                    self.logger.log({"train/loss": train_loss, "train/acc": train_accuracy, "elo": model_elo})

                if (epoch+1) % save_term == 0:
                    save_path = self.save_base_path / f"ckpt/epoch-{iteration}-{epoch}.pkl"
                    torch.save({"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}, save_path)
                
                print(f"{iteration}/{epoch}th EPOCH DONE!! ---- Train: {train_loss}")

        
def push_and_pull(train_data, opt, lnet, gnet, res_queue, g_elo, total_step, scenario_turn, name, save_base_path):
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
                
                num_correct += ((value>0.5)==win).sum()
                num_samples += value.size(0)
            
            save_result(X[0], Y[0], policy[0], save_base_path, nrow=nrow, ncol=ncol, epoch=epoch)
        
        return sum(_loss) / len(_loss), num_correct / num_samples

    training_loss, training_acc = shared_training_one_epoch(train_loader, lnet, total_step, nrow=nrow, ncol=ncol, save_base_path=save_base_path)

    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    lnet.load_state_dict(gnet.state_dict())

    res_queue.put((training_loss, training_acc, g_elo.value, total_step, scenario_turn, name))
    return

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_elo, res_queue, name, save_base_path):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_elo, self.res_queue = global_elo, res_queue
        self.gnet, self.opt = gnet, opt
        self.save_base_path = save_base_path

        # self.lnet = load_base(game_info=game_info, first_channel=2, device=device, ckp_path=ckp)
        self.lnet = NewPolicyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=0.5).to(device)
        agent = EGreedyAgent(model=self.lnet, device=device, n_to_win=n_to_win, prev=False, with_history=False)
        self.updated = []
        self.mcts_graph = GGraph()
        self.root_node = MCTSNode(state=zero_state, turn=0, parent=None, mcts_graph=self.mcts_graph, agent=agent)
        self.mcts_graph.root(root_node=self.root_node)

    def run(self):
        total_step = 1
        while True:
            self.mcts_graph.init()
            self.root_node.init()
            self.mcts_graph.root(root_node=self.root_node)
            
            for scenario_turn in range(max_turn):
                print(f"Simulating {self.name}")
                self.lnet.eval()
                self.root_node.simulate()

                updated = self.root_node.updated
                train_data = get_train_data(updated, mcts_graph=self.mcts_graph)

                # update global and assign to local net
                push_and_pull(train_data, self.opt, self.lnet, self.gnet, self.res_queue, self.g_elo, total_step, scenario_turn, self.name, self.save_base_path)

            total_step += 1

            if nrow == 20 and ncol == 20 and n_to_win == 5:
                self.g_elo.value = ELO(challenger_elo=self.g_elo.value, critic_elo=base_elo, challenger=self.gnet, total_play=TOTAL_ELO_SIM, game_info=game_info, device=device, op_ckp=ckp)
                self.res_queue.put([self.g_elo.value])

def main(logger, save_base_path, gnet, opt):
    global_elo, res_queue = mp.Value('d', 100.), mp.Queue()
    # total_workers = mp.cpu_count()-1
    total_workers = 6
    workers = [Worker(gnet=gnet, opt=opt, global_elo=global_elo, name=i, res_queue=res_queue, save_base_path=save_base_path) for i in range(total_workers)]
    print(f"Total {len(workers)} workers.")
    [w.start() for w in workers]
    while True:
        r = res_queue.get()

        if r == None:
            break    
        elif len(r) == 0:
            logger.log({"elo": r[0]})
        
        train_loss, train_accuracy, current_elo, total_step, scenario_turn, name  = r

        if (scenario_turn+1) % save_term == 0:
            print(scenario_turn)
            save_path = save_base_path / f"ckpt/epoch-{total_step}-{scenario_turn}-{name}.pkl"
            print(f"Save in {save_path}")
            torch.save({"model": gnet.state_dict(), "optim": opt.state_dict()}, save_path)
            
        if log:    
            logger.log({"train/loss": train_loss, "train/acc": train_accuracy})
    [w.join() for w in workers]

def normal_train(logger, save_base_path, gnet, opt, args):
    agent = EGreedyAgent(model=gnet, device=device, n_to_win=n_to_win, prev=False, with_history=False)

    zero = Zero(model=gnet, optimizer=opt, agent=agent, args=args, device=device, save_base_path=save_base_path, logger=logger)

    zero.learn()

if __name__ == "__main__":
    mp.set_start_method('spawn')

    args = {"C": 2, "num_searches": 60, "num_iteration": 100, "num_self_play_iterations": 500, "num_epochs": 5}

    # channels = [2, 64, 128, 256, 128, 64, 32, 1]
    # channels = [2, 64, 128, 64, 1]
    channels = [3, 64, 128, 256, 128, 64, 32, 1]
    dropout = 0.2
    model = NewPolicyValueNet(nrow=nrow, ncol=ncol, channels=channels, dropout=dropout)
    model.to(device)
    # model = load_base(game_info, first_channel=2, device=device, ckp_path=ckp)
    model.share_memory()

    # agent = PytorchAgent(model=model, device=device, n_to_win=n_to_win, with_history=False)

    # optimizer_ckp = torch.load(ckp, map_location=torch.device(device))["optim"]
    # optimizer.load_state_dict(optimizer_ckp)
    if is_mp:
        optimizer = SharedAdam(model.parameters(), lr=learning_rate, betas=(0.92, 0.999))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.92, 0.999))
    total_parameters = get_total_parameters(model)
    print(total_parameters)

    save_base_path = Path(f"./tmp/history_{int(time.time()*1000)}")
    Path("./tmp").mkdir(exist_ok=True)
    save_base_path.mkdir(exist_ok=True)
    (save_base_path / "ckpt").mkdir(exist_ok=True)
    (save_base_path / "evalresults").mkdir(exist_ok=True)
    (save_base_path / "trainresults").mkdir(exist_ok=True)

    if log:
        logger = wandb.init(
            project="AlphaGomu"
        )
    else:
        logger = None

    if is_mp:
        main(logger, save_base_path, gnet=model, opt=optimizer)
    else:
        normal_train(logger, save_base_path, gnet=model, opt=optimizer, args=args)