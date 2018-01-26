# HOPF

Higher Order Propagation Framework 

The Frameworks provides different kernels for performing Semi-supervised Node classification.

Few Available Kernels: </br>
1> Graph Convolutional Networks (GCN)  </br>
2> Node Information Preserving Kernel </br>
3> Gated GCN </br>


view parser.py to </br>
1> Specify the kernel </br>
2> Specify the number of hops </br>
3> Specify the node and neighbor features </br>
4> Specify Folds and Percentages to run </br>
5> Add Skip Connections </br>
6> Specify Gated GCN Models </br>
7> Specify Datset, batch size, learning rate, dropout, dimensions </br>
8> Specify weighted cross entropy loss </br>
9> Partial Neighborhood </br>
.. etc </br>
</br>
View script_cora.sh and run_cora.py to know how to run multiple kernels in parallel  