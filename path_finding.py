# This case isn't unusual.
# It's quite excited to combine Q and * Q*.

class Node():
    def __init__(self, move, parent):
        self.move, self. parent, self.children = move, parent, []
        self.wins, self.visits = 0,0


# To find optimal decision we should follow up the tree.
def dijkstra():
    