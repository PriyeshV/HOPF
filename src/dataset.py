from src.utils.utils import *
import tensorflow as tf
import time

# TODO We should move all numpy data to TF data with device set to CPU
# Should we remove edges to unlabeled nodes ? That will improve performance
class Dataset:
    def __init__(self, config):
        self.config = config

        t0 = time.time()
        self.adjmat, features, self.targets, mask, nodes, self.wce, self.config.multilabel = self.load_data(config)
        print("Dataset loaded in: ", time.time()-t0)

        self.train_mask, self.val_mask, self.test_mask = mask
        self.train_nodes, self.val_nodes, self.test_nodes = nodes

        self.config.n_train = self.train_nodes.shape[0]
        self.config.n_val = self.val_nodes.shape[0]
        self.config.n_test = self.test_nodes.shape[0]
        self.config.n_nodes = self.adjmat.shape[0]

        # Get Degree
        self.degrees = np.array(self.adjmat.sum(1))

        # Some pre processing
        if self.config.drop_features > 0:
            n_features = features.shape[1]
            drop_features = 1 - self.config.drop_features
            ids = np.random.choice(range(n_features), size=int(n_features * drop_features))
            features = features[:, ids]

        # TODO Degree log issue
        self.features = features
        if self.config.dataset_name != 'reddit' and self.config.dataset_name != 'wiki':
            self.features = preprocess_features(self.features, self.degrees)

        self.config.n_nodes, self.config.n_features = self.features.shape
        self.config.n_labels = self.targets.shape[1]

        # self.print_statistics()


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

    def get_connected_nodes(self, nodes):
        # nodes are list of positions and not mask
        if self.config.max_depth == 0:
            return nodes

        neighbors = set(nodes)
        neigh_last = set(neighbors)
        for _ in range(self.config.max_depth):
            new_k = set()
            for n in neigh_last:
                new_k.update(self.adjlist[n])
            neigh_last = new_k - neighbors
            neighbors.update(neigh_last)

        return np.append(nodes, np.asarray(list(neighbors-set(nodes)), dtype=int))

    def load_data(self, config):

        # Load features | Attributes
        features = np.load(config.paths['features']).astype(np.float)
        if config.sparse_features:
            features = sp.csr_matrix(features)

        # Load labels
        labels = np.load(config.paths['labels'])

        # Load train, test and val masks
        prefix = path.join(config.paths['data'], config.label_type, config.train_percent, config.train_fold)
        test_mask = np.load(path.join(prefix, 'test_ids.npy'))
        train_mask = np.load(path.join(prefix, 'train_ids.npy'))
        val_mask = np.load(path.join(prefix, 'val_ids.npy'))

        train_nodes = np.where(train_mask)[0]
        val_nodes = np.where(val_mask)[0]
        test_nodes = np.where(test_mask)[0]

        # Load adjacency matrix - convert to sparse if not sparse # if not sp.issparse(adj):
        adjmat = sio.loadmat(config.paths['adjmat'])['adjmat']
        # if config.transductive:
        #     print('data processing - removing edges b/w train and (test + val)')
        #     # test_val_nodes = np.hstack([test_nodes, val_nodes])
            #
            # for node in train_nodes:
            #     start = adjmat.indptr[node]
            #     end = adjmat.indptr[node + 1]
            #     if start - end == 0:
            #         continue
            #     neighbors = adjmat.indices[start:end]
            #     remove_pos = np.in1d(neighbors, test_val_nodes)
            #     n_removes = np.count_nonzero(remove_pos)
            #     if n_removes == 0:
            #         continue
            #     neighbors[remove_pos] = np.zeros(n_removes)
            #     adjmat.data[start:end] = neighbors
            #
            # for node in test_val_nodes:
            #     start = adjmat.indptr[node]
            #     end = adjmat.indptr[node + 1]
            #     if start - end == 0:
            #         continue
            #     neighbors = adjmat.indices[start:end]
            #     remove_pos = np.in1d(neighbors, train_nodes)
            #     n_removes = np.count_nonzero(remove_pos)
            #     if n_removes == 0:
            #         continue
            #     neighbors[remove_pos] = np.zeros(n_removes)
            #     adjmat.data[start:end] = neighbors
            #
            # print('edge removal over')
            # print(np.count_nonzero(np.where(adjmat.sum(1)) == 0)[0])
            # adjmat.eliminate_zeros()

        graph = nx.from_scipy_sparse_matrix(adjmat)
        # Makes it undirected graph it CSR format
        adjmat = nx.adjacency_matrix(graph)
        self.adjlist = graph.adjacency_list()

        # .indices attribute should only be used on row slices
        if not isinstance(adjmat, sp.csr_matrix):
            adjmat = sp.csr_matrix(adjmat)

        # Get weights for weighted cross entropy;
        wce = get_wce(labels, train_mask, val_mask, config.wce)

        # check whether the dataset has multilabel or multiclass samples
        multilabel = np.sum(labels) > np.shape(labels)[0]

        return adjmat, features, labels, (train_mask, val_mask, test_mask), (train_nodes, val_nodes, test_nodes), wce, multilabel

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
            t0 = time.time()
            start = batch_id * batch_size
            end = np.min([(batch_id+1) * batch_size, n_nodes])
            curr_bsize = end - start
            batch_density = curr_bsize / n_nodes

            node_ids = nodes[start:end]
            n_node_ids = np.shape(node_ids)[0]

            connected_nodes = self.get_connected_nodes(node_ids)
            n_conn_nodes = connected_nodes.shape[0]

            adjmat = self.adjmat[connected_nodes, :].tocsc()[:, connected_nodes]
            if self.config.neighbors[0] != -1:
                degrees = adjmat.sum(axis=0)
                degrees = np.squeeze(np.asarray(degrees))
            else:
                degrees = self.degrees[connected_nodes]
                degrees = np.squeeze(degrees)

            adjmat = adjmat.tocoo()
            a_indices = np.mat([adjmat.row, adjmat.col]).transpose()

            # Features
            features = self.features[connected_nodes, :]
            if not self.config.featureless:
                if self.config.add_noise > 0:
                    print('Inside add Noise')
                    if self.config.sparse_features:
                        features.data += np.random.normal(0, self.config.add_noise, np.shape(nnz_features))
                    else:
                        features += np.random.normal(0, self.config.add_noise, np.shape(features))
            else:
                print('Inside drop features')
                rows = np.array(range(curr_bsize))
                cols = np.array(range(curr_bsize))
                data = np.ones(curr_bsize)
                features = sp.csr_matrix((rows, cols, data), shape=[curr_bsize, self.config.n_nodes])
                # nnz_features = features.count_nonzero()

            if self.config.sparse_features:
                nnz_features = np.array([features.count_nonzero()], dtype=np.int64)
                features = features.tocoo()
                f_indices = np.mat([features.row, features.col]).transpose()
                f_shape = features.shape
                features = features.data
            else:
                f_indices = 0
                nnz_features = 0
                f_shape = np.shape(features)

            targets = self.targets[node_ids, :]

            # Initilaize mask for outputs
            # TODO remove mask as the nodes are ordered - make changes in __main__.py
            mask = np.zeros(n_conn_nodes, dtype=np.bool)
            mask[:curr_bsize] = True

            # print("Batch generated in: ", time.time()-t0)
            yield mask, degrees, n_conn_nodes, n_node_ids, batch_density, f_indices, features, f_shape, \
                nnz_features, targets, node_ids, connected_nodes, a_indices, adjmat.data, adjmat.shape
