# HOPF

Higher Order Propagation Framework 

The Frameworks provides different kernels for performing Semi-supervised Node classification.

Few Available Kernels: </br>
1> Graph Convolutional Networks (GCN)  </br>
2> Node Information Preserving Kernel </br>
3> Gated GCN </br>


view parser.py to
1> Specify the kernel
2> Specify the number of hops
3> Specify the node and neighbor features 
4> Specify Folds and Percentages to run
5> Add Skip Connections
6> Specify Gated GCN Models
7> Specify Datset, batch size, learning rate, dropout, dimensions
8> Specify weighted cross entropy loss
9> Partial Neighborhood
.. etc

View script_cora.sh and run_cora.py to know how to run multiple kernels in parallel 