import pickle
import networkx as nx
import scipy.io as sio

datasets = ['cora', 'citeseer', 'pubmed', 'mlgene', 'facebook', 'amazon',
            'reddit_ind', 'ppi_gs', 'ppi_gs_trans']

for dataset in datasets:
    file_name = dataset + '/adjmat.mat'
    adjmat = sio.loadmat(file_name)['adjmat']

    graph = nx.from_scipy_sparse_matrix(adjmat)
    adjlist = graph.adjacency_list()

    save_name = dataset + '/adjlist.pkl'
    with open(save_name, "wb") as fp:
        pickle.dump(adjlist, fp)

    with open(save_name, "rb") as fp:
        adjlist = pickle.load(fp)

    print(adjlist[0])
