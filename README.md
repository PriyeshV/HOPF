# HOPF: Higher Order Propagation Framework for Deep Collective Classification.

A modular framework for node classification in graphs with node attributes. 

#### Paper links

Vijayan, Priyesh, Yash Chandak, Mitesh M. Khapra, and Balaraman Ravindran. "HOPF: Higher Order Propagation Framework for Deep Collective Classification." arXiv preprint arXiv:1805.12421 (2018). 

Vijayan, Priyesh, Yash Chandak, Mitesh M. Khapra, and Balaraman Ravindran. "Fusion Graph Convolutional Networks." arXiv preprint arXiv:1805.12528 (2018).


#### Installation
This is a TensorFlow 1.3 implementation written in Python 3.5. <br>
All required packages to run HOPF is provided in installations.sh

### Contents

The Frameworks provides different kernels for performing Semi-Supervised learning for Node classification.

###### Graph Kernels:
    1> Graph Convolutional Networks (GCN) Kernel 
    2> Node Information Preserving Kernel 
    3> Fusion GCN Kernel

###### Polynomial bases:
        1> Binomial basis:
            - Supports Node only and Neighbor only baselines
            - Skip connection supported for Graph Convolutions 
            - Supported kernels: simple | kipf 
            - Fusion model available: binomial_fusion
        2> Chebyshev basis:
            - Skip connection internally turned Off (as it will change basis)
            - Supported kernels: chebyshev (Default internally)
        3> Krylov basis:
            - Skip connection internally turned Off (as it will change basis)
            - Supported kernels: simple | kipf (Kipf to be preferred)

### How to run

You can specify the following model, dataset and training parameters. 
For choices of arguments and additional parameters refer 'parser.py'.

Model Parameters 

    1> propagation model with 'propModel' | 'propagation' is used in HOPF paper; other basis/models can be chosen  
    2> graph kernel with 'aggKernel' | 'simple' is used in NIP Kernel
    3> number of hops with 'max_depth' 
    4> node features with 'node_features' | 'x' is used in NIP Kernel
    5> neighbor features with 'neighbor_features' | 'h' is used in NIP Kernel
    6> layer dimensions with 'dims'
    7> skip Connections with 'skip_connectons' 
    8> shared node and neeighbor weights with 'shared_weights' | 0: no shared weights in NIP Kernel
    9> number of HOPF iterations with 'max_outer' | 5: for I-NIP model
    10> number of neighbors at each layer with 'neighbors'
    ...

Dataset details
    
    1> dataset directory with 'dataset' 
    2> labeled percentage with 'percents'
    3> folds to run with 'folds'
    ...

Training details

    1> drop learning rate with patience based stopping criteria with 'drop_lr'
    2> weighted cross entropy loss with 'wce'
    ...
    

Usage:

    cd HOPF/src/
    export PYTHONPATH='../'
    python __main__.py --dataset amazon --propModel binomial --aggKernel kipf
    
view script_cora.sh and run_cora.py to run multiple kernels in parallel across GPUs.  

### Code Structure
    HOPF/
    ├── Datasets                               
    │   ├── amazon
    │   │   ├── adjmat.mat
    │   │   ├── features.npy
    │   │   ├── labels.npy
    │   │   ├── labels_random
    │   │   │   ├── 10
    │   │   │   │   ├── 1
    │   │   │   │   │   ├── test_ids.npy
    │   │   │   │   │   ├── train_ids.npy
    │   │   │   │   │   └── val_ids.npy
    │   │   │   │   ├── ..
    │   │       ├── 20
    │   │       │   ├── ...
    │   │       ...
    │   ├── cora
    │   └── ...
    ├── Experiments                             # Log and Outputs
    │   └── 5|10|13:56:18                           # Timestamp
    │       └── cora                                # Dataset
    │           └── simple                          # Kernel
    │               └── Default                     
    │                   └── 10                      # Labeled %
    │                       └── 2
    │                           ├── Checkpoints
    │                           ├── Embeddings
    │                           ├── Logs
    │                           └── Results
    ├── src                                         # src code
    │   ├── __main__.py                             # Main Train file
    │   ├── cells               
    │   │   └── lstm.py
    │   ├── config.py
    │   ├── dataset.py
    │   ├── layers
    │   │   ├── batch_norm.py
    │   │   ├── dense.py
    │   │   ├── fusion_attention.py
    │   │   ├── fusion_weighted_sum.py
    │   │   ├── graph_convolutions
    │   │   │   ├── chebyshev_kernel.py
    │   │   │   ├── kernel.py
    │   │   │   ├── kipf_kernel.py
    │   │   │   ├── maxpool_kernel.py
    │   │   │   └── simple_kernel.py
    │   │   └── layer.py
    │   ├── losses
    │   │   └── laplacian_regularizer.py
    │   ├── models
    │   │   ├── binomial.py
    │   │   ├── binomial_fusion.py
    │   │   ├── chebyshev.py
    │   │   ├── krylov.py
    │   │   ├── krylov2.py
    │   │   ├── model.py
    │   │   ├── model_old.py
    │   │   ├── propagation.py
    │   │   └── propagation_fusion.py
    │   ├── parser.py
    │   ├── run.py
    │   ├── run_cora.py
    │   ├── script_cora.sh
    │   ├── tabulate_results.py
    │   └── utils
    │       ├── inits.py                                # intitalizers
    │       ├── metrics.py                              # metrics
    │       ├── utils.py                                # numerous utilaries 



Code Traversal


    parser.py   --- gets arguments
    config.py   --- loads arguments and sets up working environment
    dataset.py  --- takes in config and loads dataset
    __main__.py --- takes in config and dataset objects
                --- connects TF Queues with dataset objects
                --- builds a model from 'models'
                    --- adds layers from 'layers'
                    --- adds ..
                --- starts (mini) batch training the model
                
                
#### Acknowledgements

A large portion of inital version of the code and some of the recent versions of this code was written in colloboration with Yash Chandak (https://yashchandak.github.io/). <br>
Certain segments of codebase was forked and inspired from Thomas Kipf (https://github.com/tkipf/gcn/).  <br>
This work was partly supported by a grant from Intel Technology India Pvt. Ltd. to Balaraman Ravindran and Mitesh M. Khapra.
I also thank my friend, Sandeep Mederametla who provided a considerable amount of AWS credits to support our experiments. 

               