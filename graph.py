#here we'll show how to visualize graphs and nodes using the methods we've built so far
from graphviz import Graph, Source

class TreeNode:
    def __init__(self, key=None,val=0,left=None,right=None):
        #this is for tree map
        self.key=key
        #this is used for all BSTs
        self.val = val
        #by default left and right child nodes are initalized to None
        self.left = left
        self.right = right

from graphviz import Graph

def visualize_tree(root, TreeMap=False):
    '''
    Visualizes a Binary Tree or a TreeMap, taking in the root node as a starting point
    
    :param root: The root node of the tree
    :param TreeMap: Boolean indicating if the tree is a TreeMap (default: False)
        if its a TreeMap, the keys are ordered, and each node stores a key and a value
    :return: A graph object
    '''
    def add_nodes_edges(node):
        if node:
            if TreeMap:
                node_label = f"Key {node.key}: Value {node.val}"
            else:
                node_label = str(node.val)
            
            g.node(str(id(node)), node_label)
            
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