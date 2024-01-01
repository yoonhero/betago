import uuid
import graphviz
from collections import defaultdict


# Naive Node for visualizing the game tree.
class Node():
    def __init__(self, name, value=0):
        self.name = name
        self.value = value
        self._uid = str(uuid.uuid4())
    
    def set(self, x): self.value = x

    @property
    def id(self): return self._uid

    def __repr__(self):
        return f"{self.name}-{self.value:.3f}"
    
    def __str__(self):
        return f"Node(name={self.name} value={self.value})"


# Game Tree for LOL.
class Graph:
    def __init__(self):
        # dictionary root-childrens
        # Tree Graph about connection between nodes.
        self.graph = defaultdict(list)
        self._data = {}

    def addEdge(self, parent:Node, children:Node):
        self.graph[parent.id].append(children.id)
        self._data[children.id] = children

    # Return children nodes.
    def vertexes(self, parent: Node) -> list[Node]:
        childrens = self.graph[parent.id]
        return [self._data[children_id] for children_id in childrens]
    
    def DFS(self, digraph, parent_uid):
        children_uids = self.graph[parent_uid]
        if children_uids.__len__() == 0:
            return

        for children_uid in children_uids:    
            children_node = self._data[children_uid]
            digraph.node(children_uid, children_node.__repr__())
            digraph.edge(parent_uid, children_uid)

            self.DFS(digraph=digraph, parent_uid=children_uid)

    def viz(self, root_node: Node):
        digraph = graphviz.Digraph()
        digraph.node(root_node.id, root_node.__repr__())
        self.DFS(digraph, root_node.id)

        return digraph