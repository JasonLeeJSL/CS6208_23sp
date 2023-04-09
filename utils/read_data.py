import networkx as nx
import pandas as pd
import os

# helper function to read data from source file

def read_graph_from_csv(file_path: str, source = 'source', target = 'target'):
    # read a graph from a csv source
    # input:
    #       file_path - the path of the csv file
    #       source - the attribute of the source column
    #       target - the attribute of the target column
    # output:
    #       G - a graph desired
    if not os.path.exists(file_path):
        raise FileExistsError("No such source file")
    data_file = pd.read_csv(file_path, sep = '\t')
    G = nx.from_pandas_edgelist(data_file, source, target, edge_attr = True, create_using=nx.Graph())
    return G
