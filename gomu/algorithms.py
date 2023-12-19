# This case isn't unusual.
# It's quite excited to combine Q and * Q*.
from collections import defaultdict

from .bot import PytorchAgent
from .game_tree import Graph, Node

# To find optimal decision we should follow up the tree.
class DijkstraAgent(PytorchAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make_game_tree(self, game_tree: Graph, board_state, depth: int, parent: tuple[int, int], my_turn: int):
        if depth == 0:
            return

        next_turn = 1 - my_turn
        selected_poses, _ = self.predict_next_pos(board_state=board_state, top_k=self.max_search_node)

        for next_pos in selected_poses:
            game_tree.addEdge(parent, next_pos)
            new_board_state = self.get_new_board_state(board_state, next_pos, my_turn)
            
            self.make_game_tree(game_tree, new_board_state, depth-1, parent=next_pos, my_turn=next_turn)

    def dijkstra(self):
        pass
        
    def forward(self, board_state, turn, max_depth_per_search=3, attempts=3):
        cur_parent = "root"
        for _ in range(attempts):
            game_tree = Graph()
            self.make_game_tree(game_tree=game_tree, board_state=board_state, depth=max_depth_per_search, parent=cur_parent, my_turn=turn)
        

def Astar():
    pass
    