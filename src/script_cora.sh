
#! /bin/sh
#! /bin/bash

python run_cora.py simple h - 0 5 0 &      # Node
sleep 2s
python run_cora.py simple x h 0 5 1 &     # Node#
sleep 2s
python run_cora.py simple h - 0 1 2 &      # Node
sleep 2s
python run_cora.py simple - h 0 1 3 # Neighbor
sleep 2s
python run_cora.py simple x h 0 1 4 &   # Simple
sleep 2s
python run_cora.py kipf x h 0 1 5 &     # Kipf GCN
sleep 2s
python run_cora.py kipf h h 1 1 6 &     # Kipf GCN
sleep 2s
python run_cora.py simple h h 1 1 7    # Simple
sleep 2s


