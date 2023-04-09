import networkx as nx
import random
# helper functions of random walk part

def get_a_walk(G: nx.graph, node, walk_length):
    # get a random walk start from a node
    # input :
    #       G - the graph
    #       node - a node in the graph as the start point
    #       walk_length - the length of each walk
    # output :
    #       a random walk path

    rw_path = []
    rw_path.append(node)

    cur_node = node
    for _ in range(walk_length):
        neighbor_nodes = list(set(G.neighbors(cur_node)) - set(rw_path))
        if len(neighbor_nodes) == 0:
            # need to be further considered
            # it was not mentioned in the paper
            # I believe that it does not matter because 'word2vec' accept different length of sentences
            # but it is a interesting thing to find that whether different walk length makes a difference?
            return rw_path
        cur_node = random.choice(neighbor_nodes)
        rw_path.append(cur_node)
    return rw_path

def get_random_walks(G: nx.graph, walk_num, walk_length):
    # get random walks
    # input :
    #       G - the graph
    #       walk_num - the number of walk to get for each vertex
    #       walk_length - the length of each walk
    # output :
    #       random walks
    walks = []
    for node in list(G.nodes()):
        for _ in range(walk_num):
            # handle walk with length < walk_length here?
            # ...
            walks.append(get_a_walk(G, node, walk_length))
    return walks

