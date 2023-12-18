import uuid
import graphviz

# Naive Node for visualizing the game tree.
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