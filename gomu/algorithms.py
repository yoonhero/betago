# This case isn't unusual.
# It's quite excited to combine Q and A*.
from collections import defaultdict
from queue import Queue
import math
import uuid
import random
from queue import PriorityQueue
from copy import deepcopy

import numpy as np
import torch
from einops import rearrange

from .bot import PytorchAgent
from .gomuku.board import GoMuKuBoard
from .game_tree import Graph, Node
from .helpers import DEBUG

class NodeForDijkstra(Node):
    def __init__(self, next_pos, value=0):
        self.next_pos = next_pos
        self.value = value
        self._uid = str(uuid.uuid4())

    def set(self, x): self.value = x

    @property
    def pos(self): return self.next_pos

    def format_pos(self): return f"{self.next_pos[0]},{self.next_pos[1]}"

    def __repr__(self):
        return f"{self.next_pos}-{self.value:.3f}"

    def __str__(self):
        return f"Node(next_pos={self.next_pos} value={self.value})"

class GraphForDijkstra(Graph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uid = uuid.uuid4()
        self.pos_connection = defaultdict(list)

    def addEdge(self, parent: NodeForDijkstra, children: NodeForDijkstra):
        super().addEdge(parent, children)
        self.pos_connection[parent.format_pos()].append(children.id)
    
    def getEndingScenarios(self):
        win = []
        lose = []
        for (key, node) in self._data.items():
            # if math.abs(node.value) == float("inf"):
            if node.value == float("inf"):
                win.append(key)
            elif node.value == float("inf"):
                lose.append(key)
                
        return win, lose

# To find optimal decision we should follow up the tree.
class DijkstraAgent(PytorchAgent):
    def __init__(self, max_depth_per_search, max_search_vertex, **kwargs):
        super().__init__(**kwargs)
        self.max_depth_per_search = max_depth_per_search
        self.max_search_vertex = max_search_vertex

    def dijkstra(self, game_tree: GraphForDijkstra, root_node: NodeForDijkstra, mode: int):
        # mode
        # 1. find the shortest path to infinite value scenario.
        # 2. find the safe path toward infinite value scenario.

        # Root's children
        childrens = game_tree.vertexes(root_node)

        # First Check are there any ending possible scenarios.
        # If there aren't existed the winning possible scenario, Just return highest value next position prediction.
        win, lose = game_tree.getEndingScenarios()

        if win.__len__() == 0:
            max_val = 0
            max_node = None
            for children in childrens:
                if children.value >= max_val:
                    max_val = children.value
                    max_node = children

            return max_node.next_pos, max_val

        if mode == 1:
            selected_key = random.choice(win)
            to_search = Queue()
            for children in childrens:
                to_search.put((children.id, children.pos, 1))

            # BFS searching
            while True:
                current_id, pos, depth = to_search.get()
                children_keys = game_tree.graph[current_id] 
                for children_key in children_keys:
                    # Return the searching result after find the shortest path to winning.
                    if children_key == selected_key:
                        if DEBUG >= 2:
                            print(f"Find the shorted path on depth {depth}")
                        return pos, float("inf")
                    
                    to_search.put((children_key, pos, depth+1))
        
        elif mode == 2:
            pass

      # Making a Game Tree
    def make_game_tree(self, parent_node: NodeForDijkstra, game_tree: GraphForDijkstra, board_state, depth: int, is_my_turn: bool, turn: int):
        heuristic_value = self.heuristic(board_state=board_state, my_turn=1-turn)
        if is_my_turn: heuristic_value = -heuristic_value
        if heuristic_value != 0: return heuristic_value

        if depth == 0:
            _, tensor_value = self.model_predict(state=board_state)
            value = tensor_value.cpu().item()

            if is_my_turn:
                value = 1 - value

            return value

        next_turn = 1 - turn
        selected_poses, cur_value = self.predict_next_pos(board_state=board_state, top_k=self.max_search_vertex)
        cur_value = cur_value.cpu().item()

        for next_pos in selected_poses:
            current_node = NodeForDijkstra(next_pos)
            game_tree.addEdge(parent_node, current_node)

            new_board_state = self.get_new_board_state(board_state, next_pos, turn)
            # GoMuKuBoard.viz(new_board_state).save(f"./tmp/history/{game_tree.uid}-{depth}-{current_node.id}.png")
            
            value = self.make_game_tree(parent_node=current_node, game_tree=game_tree, board_state=new_board_state, depth=depth-1, is_my_turn=not is_my_turn, turn=next_turn)
            current_node.set(value)
        
        if not is_my_turn:
            cur_value = 1-cur_value
        return cur_value

        
    def forward(self, board_state, turn):
        root_node = NodeForDijkstra((-1, -1))

        game_tree = GraphForDijkstra()
        self.make_game_tree(game_tree=game_tree, board_state=board_state, depth=self.max_depth_per_search, is_my_turn=True, parent_node=root_node, turn=turn)

        if DEBUG >= 2:
            game_tree.viz(root_node).view()

        next_pos, value = self.dijkstra(game_tree=game_tree, root_node=root_node, mode=1)

        return next_pos, value



class QstarAgent(PytorchAgent):
    def __init__(self, max_depth, max_vertexs, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.max_vertexs = max_vertexs
        self.maximum_affordable = int(max_vertexs*max_depth * 0.8)

    def best_choice(self, board_state, turn, history):
        # this value is transition cost.
        next_poses, value = self.predict_next_pos(board_state, top_k=1, best=True, history=history)
        next_pos = next_poses[0]
        new_state = self.get_new_board_state(board_state=board_state, next_pos=next_pos, my_turn=turn)

        return new_state, value.cpu().item(), next_pos
    
    def numpy_to_key(self, board: np.ndarray):
        # return "".join([str(int(stone)) for stone in board.tolist()])
        return board.tobytes()
    
    def get_q_with_batch(self, board_state, turn, next_poses, history):
        _, ncol, nrow = board_state.shape
        B = len(next_poses)
        board_state = self.preprocess_state(board_state)
        board_state = board_state.repeat(B, 1, 1, 1)
        
        # board_state = rearrange(board_state, "B C H W -> C B H W")
        # board_state[turn, next_poses] = 1
        for i in range(B):
            board_state[i, turn, next_poses[i][1], next_poses[i][0]] = 1
        # board_state = rearrange(board_state, "")
        if self.with_history:
            tensor_history = self.make_history_state(history, nrow=nrow, ncol=ncol)
            tensor_history = tensor_history.repeat(B, 1, 1, 1)
            board_state = torch.cat([tensor_history, board_state], dim=0)

        _, value = self.model(board_state)
        return value.cpu().detach().tolist()
    
    def forward(self, board_state, turn, gamma=0.9, **kwargs):
        total_stone = (board_state != 0).sum()

        if total_stone <= 6:
            return super().forward(board_state, **kwargs)
        
        # Fine the Optimal Choice based on astar algorihtm
        open = PriorityQueue()
        closed = defaultdict(lambda: None)
        valuemap = defaultdict(lambda: float("inf"))        
        my_turn = turn
        opposite_turn = 1 - turn

        history = self.history

        my_initial_best_choices, _ = self.predict_next_pos(board_state, history=history, top_k=self.max_vertexs, best=True)
        for i, my_initial_best_choice in enumerate(my_initial_best_choices):
            initial_item = (0, board_state, my_initial_best_choice, 0, 0, i, deepcopy(history))
            open.put(initial_item)
        
        count = 0
        
        # Transition Cost = Next Mover's Expected Value?
        while not open.empty():
            _, state, action, prev_state_cost, depth, ith, _history = open.get()

            # restrict the max searching depth for no solution case.
            if depth >= self.max_depth:
                count += 1
                if count <= self.maximum_affordable:
                    continue

                if DEBUG >= 2:
                    print("STOP Searching Cause Current Depth Max Depth Setting.")
                return super().forward(board_state, **kwargs)
            
            new_state = self.get_new_board_state(state, next_pos=action, my_turn=my_turn)
            # Exit the loop when found the desired board state.
            if self.heuristic(board_state=new_state, my_turn=my_turn) > 0:
                break

            newnew_state, current_cost, best_next_pose = self.best_choice(board_state=new_state, turn=opposite_turn, history=_history)
            # _history.append(best_next_pose)
            # Prevent ignoring the ending scenario
            if self.heuristic(board_state=newnew_state, my_turn=opposite_turn) > 0:
                continue

            state_cost = prev_state_cost + current_cost

            str_state = self.numpy_to_key(state)
            str_newnew_state = self.numpy_to_key(newnew_state)
            if str_state not in closed.keys() or state_cost < valuemap[str_newnew_state]:
                closed[str_newnew_state] = state
                valuemap[str_newnew_state] = state_cost
                next_poses, _ = self.predict_next_pos(newnew_state, top_k=self.max_vertexs, best=False, history=_history)
                qs = self.get_q_with_batch(board_state=newnew_state, next_poses=next_poses, turn=turn, history=None)

                for i, next_pos in enumerate(next_poses):
                    q = qs[i][0]
                    heuristic_cost = (1-q)
                    # EMA
                    final_cost = gamma * prev_state_cost + current_cost + heuristic_cost
                    item = (final_cost, newnew_state, next_pos, state_cost, depth+1, ith, None)
                    open.put(item)
        
        # Reconstruct the path
        # while True:
        #     state = closed[state]
        #     if state == None:
        #         break
        #     # col, row = cur_action
        #     # state[cur_turn, row, col] = 0
        next_pos = my_initial_best_choices[ith]
        if DEBUG >= 2:
            GoMuKuBoard.viz(board_state=state).show()
            print("Q* Searching Finished!!")
            print(f"Predicted Next Position: {next_pos}")
        return next_pos, 1