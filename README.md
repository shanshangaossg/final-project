We need rdkit to run this code.
Install rdkit: https://www.rdkit.org/docs/Install.html

The ideas for designing node and edge features in this project were inspired by: https://www.researchgate.net/figure/Descriptions-of-node-and-edge-features_tbl1_339424976

Usage
1. Preprocess the Data:
   Prepare the raw molecular datasets by running the preprocessing script: dataset.py
2. Construct the Model:
   Construct the Model using model.py
4. Train the Model:
   Train and evaluate one of the GNN models (GAT, GATv2, GCN) using train.py

