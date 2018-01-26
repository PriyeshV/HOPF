
#! /bin/sh
#! /bin/bash

# Format
# python run_cora.py simple/kipf_kernels node_features neighbors_features no_outer_epochs GPU_id

# Node Attribute Only Classifier
python run_cora.py simple h - 0 1 2 &      # Node
sleep 2s

# Neighbor only Classifier
python run_cora.py simple - h 0 1 3 # Neighbor
sleep 2s

# Node Information preserving (NIP) model with unnormalized laplacian kernel
python run_cora.py simple x h 0 1 4 &   # Simple
sleep 2s

# Node Information preserving model with Thomas Kipf's GCN style weighted combination
python run_cora.py kipf x h 0 1 5 &     # Kipf GCN
sleep 2s

# Thomas Kipf's GCN weights but HOPF version
python run_cora.py kipf h h 1 1 6 &     # Kipf GCN
sleep 2s

# Mean kernel
python run_cora.py simple h h 1 1 7    # Simple
sleep 2s

# Semi-supervised ICA
python run_cora.py simple h - 0 5 0 &      # Node
sleep 2s

# Semi-supervised Iterative NIP Kernels
python run_cora.py simple x h 0 5 1 &     # Node#

