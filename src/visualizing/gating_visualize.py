import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

# Load data and scores
adjmat_path = '../../Datasets/citeseer/adjmat.mat'
labels_path = '../../Datasets/citeseer/labels.npy'
gating_path = './citeseer-1-3_gating_scores.npy'
label = 0

adjmat = sio.loadmat(adjmat_path)['adjmat']
labels = np.load(labels_path)
n_labels = np.shape(labels)[1]
scores = np.load(gating_path)

print(np.shape(adjmat), np.shape(scores))

# Create nx graph
graph = nx.from_scipy_sparse_matrix(adjmat)
adjmat = nx.adjacency_matrix(graph)  # Makes it undirected graph it CSR format

# Set color based on scores
colors = {0:{'r' : 0, 'g' : 0, 'b' : 255},
          1:{'r' : 0, 'g' : 255, 'b' : 0},
          2:{'r' : 255, 'g' : 0, 'b' : 0} ,
          3:{'r' : 0, 'g' : 255, 'b' : 255} ,
          4:{'r' : 255, 'g' : 0, 'b' : 255} ,
          5:{'r' : 255, 'g' : 255, 'b' : 0} ,
            6: {'r': 0, 'g': 0, 'b': 127},
            7: {'r': 0, 'g': 127, 'b': 0},
            8: {'r': 127, 'g': 0, 'b': 0},
            9: {'r': 0, 'g': 127, 'b': 127},
            10: {'r': 127, 'g': 0, 'b': 127},
            11: {'r': 127, 'g': 127, 'b': 0}}



for id in range(adjmat.shape[0]):
    # Depth importance based coloring
    # pos = np.argmax(scores[id, :, label])
    pos = np.argmax(np.sum(scores[id], axis=1))
    graph.node[id] = {'viz': {'color': colors[pos]}}

    # Label based coloring
    # pos = np.argmax(labels[id])
    # graph.node[id] = {'viz': {'color': colors[pos]}}


# Save in Gephi format
nx.write_gexf(graph, 'graph_colored-citeseer.gexf')#, version="1.2draft")