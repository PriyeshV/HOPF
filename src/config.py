import tensorflow as tf
import sys
from yaml import dump
from os import path
from src.utils import utils
import numpy as np
import importlib


class Config(object):
    """ Helps create a configuration object which will
        - contains all arguments from parser
        - set some additional default configurations """

    def __init__(self, args):

        # SET UP PATHS
        self.paths = dict()
        self.paths['root'] = '../'

        self.paths['datasets'] = path.join(self.paths['root'], 'Datasets')
        self.paths['experiments'] = path.join(self.paths['root'], 'Experiments')
        self.dataset_name = args.dataset
        self.paths['experiment'] = path.join(self.paths['experiments'], args.timestamp, self.dataset_name, args.aggKernel, args.folder_suffix)

        # Parse training percentages and folds
        self.train_percents = args.percents.split(',')
        self.train_folds = args.folds.split(',')
        for perc in self.train_percents:
            self.paths['perc' + '_' + perc] = path.join(self.paths['experiment'], perc)
            for fold in self.train_folds:
                suffix = '_' + perc + '_' + fold
                path_prefix = [self.paths['experiment'], perc, fold, ]
                self.paths['logs' + suffix] = path.join(*path_prefix, 'Logs/')
                self.paths['ckpt' + suffix] = path.join(*path_prefix, 'Checkpoints/')
                self.paths['embed' + suffix] = path.join(*path_prefix, 'Embeddings/')
                self.paths['results' + suffix] = path.join(*path_prefix, 'Results/')

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'experiments', 'datasets']:
                utils.create_directory_tree(str.split(val, sep='/')[:-1])

        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False, explicit_start=True)
        self.paths['data'] = path.join(self.paths['datasets'], self.dataset_name)
        self.paths['labels'] = path.join(path.join(self.paths['data'], 'labels.npy'))
        self.paths['features'] = path.join(path.join(self.paths['data'], 'features.npy'))
        self.paths['adjmat'] = path.join(path.join(self.paths['data'], 'adjmat.mat'))

        # -------------------------------------------------------------------------------------------------------------

        # Hidden dimensions
        self.dims = list(map(int, args.dims.split(',')[:args.max_depth]))
        if len(self.dims) < args.max_depth:
            sys.exit('#Hidden dimensions should match the max depth')

        # Propogation Depth
        self.max_depth = args.max_depth
        if self.max_depth == 0:
            sys.exit('Error: Invalid value. Supported range > 0  \n For 0th hop, set neighbor_features=- and self.max_depth=k for k layer deep FFN')

        # Drop nodes or edges
        self.drop_edges = args.drop_edges

        # Subset of neighbors to consider at max
        self.neighbors = args.neighbors.split(',')
        for i in range(len(self.neighbors)):
            if self.neighbors[i] == 'all':
                self.neighbors[i] = '-1'
        self.neighbors = np.asarray(self.neighbors, dtype=int)
        if self.neighbors.shape[0] < args.max_depth:
            # Extend as -1 is no information provided, i.e take all neighbors at that depth
            # diff = args.max_depth - self.neighbors.shape[0]
            # self.neighbors = np.hstack((self.neighbors, [-1] * diff))
            sys.exit('Neighbors argument should match max depth: ex: all,1')

        if args.drop_edges != 0 and self.neighbors[0] != -1:
            sys.exit('Can not have drop edges and neighbors flag set at the same time')

        # GPU
        self.gpu = args.gpu

        # Data sets
        self.label_type = args.labels

        # Weighed cross entropy loss
        self.wce = args.wce

        # Retrain
        self.retrain = args.retrain

        # Metrics to compute
        self.metric_keys = ['accuracy', 'micro_f1', 'macro_f1', 'bae']

        # Batch size
        if args.batch_size == -1:
            self.queue_capacity = 1
        else:
            self.queue_capacity = 5
        self.batch_size = args.batch_size

        # Dropouts
        self.drop_in = args.drop_in
        # self.drop_out = args.drop_out
        # self.drop_conv = args.drop_conv
        self.drop_conv = self.drop_in
        self.drop_out = self.drop_in
        # self.drop_out = args.drop_out

        # Data pertubation
        self.drop_features = args.drop_features
        self.add_noise = args.add_noise

        # Number of steps to run trainer
        self.max_outer_epochs = args.max_outer
        self.max_inner_epochs = args.max_inner

        # Save summaries
        self.summaries = args.summaries

        # Validation frequence
        self.val_epochs_freq = args.val_freq  # 1
        # Model save frequency
        self.save_epochs_after = args.save_after  # 0

        # early stopping hyper parametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.learning_rate = args.lr
        self.drop_lr = args.drop_lr

        # optimizer
        self.l2 = args.l2
        if args.opt == 'adam':
            self.opt = tf.train.AdamOptimizer
        elif args.opt == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer
        elif args.opt == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer
        else:
            raise ValueError('Undefined type of optmizer')

        # -------------------------------------------------------------------------------------------------------------

        # Set Model
        self.kernel_class = getattr(importlib.import_module("src.layers.graph_convolutions."+args.aggKernel+"_kernel"), "Kernel")
        self.prop_class = getattr(importlib.import_module("src.models."+args.propModel), "Propagation")
        self.prop_model_name = args.propModel
        self.kernel_name = args.aggKernel

        # Sparse Feature settings
        self.sparse_features = args.sparse_features
        if not self.sparse_features and self.dataset_name in ['cora', 'citeseer', 'amazon', 'facebook', 'cora_multi', 'movielens',
                                    'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs']:
            self.sparse_features = True
            print('Sparse Features turned on forcibly!')
        elif self.dataset_name in ['wiki', 'reddit']:
            self.sparse_features = False

        # Node features
        self.features = ['x', 'h']
        if args.node_features == '-':
            self.n_node_features = 0
            self.node_features = ''
        else:
            self.node_features = args.node_features.split(',')
            self.n_node_features = len(self.node_features)

        if args.neighbor_features == '-':
            self.n_neigh_features = 0
            self.neighbor_features = ''
        else:
            self.neighbor_features = args.neighbor_features.split(',')
            self.n_neigh_features = len(self.neighbor_features)

        if self.n_node_features == 0 and self.n_neigh_features == 0:
            sys.exit("Both node and neigh features can't be empty")
        else:
            if self.n_node_features != 0 and np.count_nonzero(np.in1d(self.node_features, self.features)) != self.n_node_features:
                sys.exit('Invalid node features. small case \'x and \'h are only valid')
            if self.n_neigh_features != 0 and np.count_nonzero(np.in1d(self.neighbor_features, self.features)) != self.n_neigh_features:
                sys.exit('Invalid neighbor features. small case \'x and \'h are only valid')

        # Loss terms
        self.loss = {}
        self.loss['l2'] = args.l2

        # Featureless
        self.featureless = args.featureless

        # Skip connections
        self.skip_connections = args.skip_connections

        if args.shared_weights == 1:
            self.shared_weights = True
        else:
            self.shared_weights = False
        if args.node_features == '-':
            self.shared_weights = False
        self.bias = args.bias

        # Outer (HOPF) iterations for iterative inference
        self.add_labels = False
        if self.max_outer_epochs > 1:
            self.add_labels = True
            if '-' in self.neighbor_features:
                self.max_depth = 1

        self.save_model = args.save_model