#here we'll show how to visualize graphs and nodes using the methods we've built so far
from graphviz import Graph, Source

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def visualize_tree(root):
    '''assumes root is the root node of a Binary Tree
    
    return a graph object'''
    def add_nodes_edges(node):
        if node:
            g.node(str(id(node)), str(node.val))
            if node.left:
                g.edge(str(id(node)), str(id(node.left)))
                add_nodes_edges(node.left)
            if node.right:
                g.edge(str(id(node)), str(id(node.right)))
                add_nodes_edges(node.right)

    g = Graph()
    g.attr(rankdir='TB')
    add_nodes_edges(root)
    
    return g