import pickle
import networkx as nx
import scipy.io as sio
from os import path

datasets = ['cora', 'citeseer', 'pubmed', 'mlgene', 'facebook', 'amazon', 'movielens', 'blogcatalog',
            'reddit_trans', 'ppi_gs_trans']

for dataset in datasets:
    print('working on dataset: ', dataset)
    file_name = path.join(dataset, 'adjmat.mat')
    if not path.exists(file_name):
        print('dataset not created :', dataset)
    adjmat = sio.loadmat(file_name)['adjmat']

    graph = nx.from_scipy_sparse_matrix(adjmat)
    adjlist = graph.adjacency_list()

    save_name = dataset + '/adjlist.pkl'
    with open(save_name, "wb") as fp:
        pickle.dump(adjlist, fp)

    with open(save_name, "rb") as fp:
        adjlist = pickle.load(fp)
