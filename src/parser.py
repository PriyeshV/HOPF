import argparse
import numpy as np
from datetime import datetime


# TODO # Negative Contraints
class Parser(object):  #
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Node attribute Aggregator
        parser.add_argument("--propModel", default='propagation', help='propagation model names',
                            choices=['propagation', 'propagation_fusion', 'krylov1', 'krylov2'
                                     'chebyshev', 'attention', 'binomial', 'binomial_fusion'])
        parser.add_argument("--aggKernel", default='kipf', help="kernel names",
                            choices=['kipf', 'simple', 'chebyshev', 'maxpool', 'add_attention', 'mul_attention'])
        parser.add_argument("--max_depth", default=2, help="Maximum path depth", type=int)

        parser.add_argument("--node_features", default='h', choices=['x', 'h', 'x,h'], help='x for nip connections')
        parser.add_argument("--neighbor_features", default='h', choices=['x', 'h', 'x,h'])

        parser.add_argument("--dims", default='64,64,8,8,8', help="Dimensions of hidden layers: comma separated")
        parser.add_argument("--skip_connections", default=True, help="output layer added", type=self.str2bool)

        parser.add_argument("--shared_weights", default=1, type=int)
        parser.add_argument("--bias", default=False, type=self.str2bool)
        parser.add_argument("--sparse_features", default=True, help="For current datasets - manually set in config.py", type=self.str2bool)
        # parser.add_argument("--featureless", default=False, help="Non-attributed graphs", type=self.str2bool)

        # Node attributes pertubation
        parser.add_argument("--add_noise", default=0, help="Add noise to input attributes", type=float, choices=np.round(np.arange(0, 1, 0.1),1))
        parser.add_argument("--drop_features", default=0, help="Range 0-1", type=float, choices=np.round(np.arange(0, 1, 0.1),1))

        # Structure pertubation
        parser.add_argument("--neighbors", default='all,all,all,all,all', help="Number of neighbors at each depth; comma separated")
        parser.add_argument("--drop_edges", default=0., help="Randomly drop edges at each depth", type=float, choices=np.round(np.arange(0, 1, 0.1), 1))

        # Dataset Details
        parser.add_argument("--dataset", default='gcn-cora', help="Dataset to evluate | Check Datasets folder",
                            choices=['cora', 'citeseer', 'wiki', 'amazon', 'facebook', 'cora_multi', 'movielens',
                                    'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs', 'reddit', 'reddit_ind'])
        parser.add_argument("--labels", default='labels_random', help="Label Sampling Type")
        parser.add_argument("--percents", default='10', help="Training percent comma separated, ex:5,10,20")
        parser.add_argument("--folds", default='1', help="Training folds comma separated")

        # NN Hyper parameters
        parser.add_argument("--batch_size", default=128, help="Batch size", type=int)
        parser.add_argument("--wce", default=False, help="Weighted cross entropy", type=self.str2bool)
        parser.add_argument("--lr", default=1e-2, help="Learning rate", type=float)
        parser.add_argument("--l2", default=1e-3, help="L2 loss", type=float)
        parser.add_argument("--opt", default='adam', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--drop_in", default=0.5, help="Dropout for input", type=float, choices=np.round(np.arange(0, 1, 0.05),2))
        # parser.add_argument("--drop_out", default=0.5, help="Dropout for Fusion", type=float,
        #                     choices=np.round(np.arange(0, 1, 0.05), 2))

        # Training parameters
        parser.add_argument("--retrain", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--verbose", default=0, help="Verbose mode", type=int, choices=[0, 1, 2])
        parser.add_argument("--save_model", default=False, type=self.str2bool)

        parser.add_argument("--max_outer", default=3, help="Maximum outer epoch", type=int)
        parser.add_argument("--max_inner", default=70, help="Maximum inner epoch", type=int)

        parser.add_argument("--drop_lr", default=True, help="Drop lr with patience drop", type=self.str2bool)
        parser.add_argument("--pat", default=30, help="Patience", type=int)
        parser.add_argument("--save_after", default=50, help="Save after epochs", type=int)
        parser.add_argument("--val_freq", default=1, help="Validation frequency", type=int)
        parser.add_argument("--summaries", default=True, help="Save summaries after each epoch", type=self.str2bool)

        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='ppigs_256_4hops', help="folder name suffix")

        # TODO Load saved model and saved argparse
        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

