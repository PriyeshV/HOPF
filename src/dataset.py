from src.utils.utils import *
import time
import pickle
import os.path
import random


# TODO should move all numpy data to TF data with device set to CPU
# TODO automatically create inductive setups from transductive data
# TODO dynamically call mini-batch prefetching codes for partial and complete neighborhood


class Dataset:
    """
        Helps create a dataset object that will
        - load dataset
        - provide dataset statistics
        - have modules to generate minibatch
        - add noise | drop edges ..
    """

    def __init__(self, config):
        self.config = config

        t0 = time.time()
        self.adjmat, features, self.targets, self.config.multilabel = self.load_data(config)
        print("Dataset loaded in: ", time.time()-t0)

        self.wce = False
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.train_nodes, self.val_nodes, self.test_nodes = None, None, None
        self.config.n_train, self.config.n_val, self.config.n_test, self.config.n_nodes = 0, 0, 0, 0

        self.degrees = np.array(self.adjmat.sum(1))

        # All datasets' feature are sparse inputs and typically 0/1 data except for reddit, which we standardize
        self.features = features
        if self.config.dataset_name != 'reddit' and self.config.dataset_name != 'wiki':
            self.features = preprocess_features(self.features, self.degrees)

        self.config.n_nodes, self.config.n_features = self.features.shape
        self.config.n_labels = self.targets.shape[1]

    def get_config(self):
        return self.config

    def get_nodes(self, node_class):
        if node_class == 'train':
            nodes, n_nodes = self.train_nodes, self.config.n_train
        elif node_class == 'val':
            nodes, n_nodes = self.val_nodes, self.config.n_val
        elif node_class == 'test':
            nodes, n_nodes = self.test_nodes, self.config.n_test
        else:
            nodes, n_nodes = np.arange(self.config.n_nodes), self.config.n_nodes
        return nodes, n_nodes

    def print_statistics(self):
        print('############### DATASET STATISTICS ####################')
        print(
            'Nodes: %d \nTrain Nodes: %d \nVal Nodes: %d \nTest Nodes: %d \nFeatures: %d \nLabels: %d \nMulti-label: %s \nMax Degree: %d \nAverage Degree: %d'\
            % (self.config.n_nodes, self.config.n_train, self.config.n_val, self.config.n_test, self.config.n_features, self.config.n_labels, self.config.multilabel, np.max(self.degrees), np.mean(self.degrees)))
        print("Cross-Entropy weights: ", np.round(self.wce, 3))
        print('-----------------------------------------------------\n')
        print('############### TRAINING STARTS --------- ####################\n\n')

    def get_connected_nodes(self, nodes):
        # nodes are list of positions and not mask
        if self.config.max_depth == 0:
            return nodes

        neighbors = set(nodes)
        neigh_last = set(neighbors)
        for h in range(self.config.max_depth):
            new_k = set()
            for n in neigh_last:
                if self.degrees[n] > 0:
                    if self.config.neighbors[h] != -1:
                        new_k.update(random.sample(self.adjlist[n], min(self.degrees[n][0], self.config.neighbors[h])))
                    else:
                        new_k.update(self.adjlist[n])
            neigh_last = new_k - neighbors
            neighbors.update(neigh_last)

        return np.append(nodes, np.asarray(list(neighbors-set(nodes)), dtype=int))

    def load_indexes(self, percent=10, fold=1):
        self.config.train_percent = percent
        self.config.train_fold = fold
        prefix = path.join(self.config.paths['data'], self.config.label_type, self.config.train_percent, self.config.train_fold)
        self.test_mask = np.load(path.join(prefix, 'test_ids.npy'))
        self.train_mask = np.load(path.join(prefix, 'train_ids.npy'))
        self.val_mask = np.load(path.join(prefix, 'val_ids.npy'))

        self.train_nodes = np.where(self.train_mask)[0]
        self.val_nodes = np.where(self.val_mask)[0]
        self.test_nodes = np.where(self.test_mask)[0]

        self.config.n_train = self.train_nodes.shape[0]
        self.config.n_val = self.val_nodes.shape[0]
        self.config.n_test = self.test_nodes.shape[0]
        self.config.n_nodes = self.adjmat.shape[0]

        # Get weights for weighted cross entropy;
        self.wce = get_wce(self.targets, self.train_mask, self.val_mask, self.config.wce)
        self.print_statistics()

    def load_data(self, config):

        # Load features | Attributes
        features = np.load(config.paths['features']).astype(np.float)
        if config.sparse_features:
            features = sp.csr_matrix(features)

        # Load labels
        labels = np.load(config.paths['labels'])

        # Load adjacency matrix - convert to sparse if not sparse # if not sp.issparse(adj):
        adjmat = sio.loadmat(config.paths['adjmat'])['adjmat']

        graph = nx.from_scipy_sparse_matrix(adjmat)
        # Makes it undirected graph it CSR format
        adjmat = nx.adjacency_matrix(graph)

        f_adjlist_name = path.join(config.paths['datasets'], config.dataset_name, 'adjlist.pkl')
        if os.path.exists(f_adjlist_name):
            print('Adjlist exist: ')
            with open(f_adjlist_name, "rb") as fp:
                self.adjlist = pickle.load(fp)
        else:
            print('does not exist')
            self.adjlist = graph.adjacency_list()
            with open(f_adjlist_name, "wb") as fp:
                pickle.dump(self.adjlist, fp)

        # .indices attribute should only be used on row slices
        if not isinstance(adjmat, sp.csr_matrix):
            adjmat = sp.csr_matrix(adjmat)

        # check whether the dataset has multilabel or multiclass samples
        multilabel = np.sum(labels) > np.shape(labels)[0]

        return adjmat, features, labels, multilabel

    def get_data(self, data):
        nodes, n_nodes = self.get_nodes(data)
        batch_size = self.config.batch_size

        if batch_size == -1:
            return nodes, n_nodes, n_nodes, 1  # nodes, n_nodes, batch_size, n_batches
        else:
            return nodes, n_nodes, min(batch_size, n_nodes), np.ceil(self.get_nodes(data)[1] / batch_size).astype(int)

    def batch_generator(self, data='train', shuffle=True):

        nodes, n_nodes, batch_size, n_batches = self.get_data(data)
        if shuffle:
            nodes = np.random.permutation(nodes)

        for batch_id in range(n_batches):
            start = batch_id * batch_size
            end = np.min([(batch_id+1) * batch_size, n_nodes])
            curr_bsize = end - start
            batch_density = curr_bsize / n_nodes

            node_ids = nodes[start:end]
            n_node_ids = np.shape(node_ids)[0]

            connected_nodes = self.get_connected_nodes(node_ids)
            n_conn_nodes = connected_nodes.shape[0]

            # Get support matrix
            adjmat = self.adjmat[connected_nodes, :].tocsc()[:, connected_nodes]
            if self.config.kernel_name == 'chebyshev':
                adjmat = get_scaled_laplacian(adjmat)

            # Adjust degree if neighbors are partially sampled
            if all(x == -1 for x in self.config.neighbors):
                degrees = self.degrees[connected_nodes]
                degrees = np.squeeze(degrees)
            else:
                degrees = adjmat.sum(axis=0)
                degrees = np.squeeze(np.asarray(degrees))

            # TF Queues does not support sparse matrix, hence we need to pass the indexes, data and shape separately
            adjmat = adjmat.tocoo()
            a_indices = np.mat([adjmat.row, adjmat.col]).transpose()

            # Features
            features = self.features[connected_nodes, :].tocoo()
            nnz_features = np.array([features.count_nonzero()], dtype=np.int64)
            f_indices = np.mat([features.row, features.col]).transpose()

            # Targets
            targets = self.targets[node_ids, :]

            # Initilaize mask for outputs
            # TODO remove mask as the nodes are ordered - make changes in __main__.py
            mask = np.zeros(n_conn_nodes, dtype=np.bool)
            mask[:curr_bsize] = True

            yield mask, degrees, n_conn_nodes, n_node_ids, batch_density, f_indices, features.data, features.shape, \
                nnz_features, targets, node_ids, connected_nodes, a_indices, adjmat.data, adjmat.shape
