device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

class GCN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pool_type="mean"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)
        self.pool_type = pool_type

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # 根据指定的池化方式进行池化
        if self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "sum":
            x = global_add_pool(x, batch)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        x = self.fc(x)
        return x

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn import Set2Set
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool
from torch_geometric.utils import subgraph
from torch_geometric.nn import BatchNorm

class GAT(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=3, dropout=0.1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim * heads) 

        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim * heads)  

        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim) 

        self.edge_fc = Linear(edge_dim, hidden_dim)
        self.pool = Set2Set(hidden_dim, processing_steps=3, num_layers=1)
        self.fc1 = Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        edge_attr = F.relu(self.edge_fc(edge_attr))

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x) 
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)  
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)  
        x = F.elu(x)

        x = self.pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x



from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import Linear, Module
import torch.nn.functional as F

from torch_geometric.nn import BatchNorm

class GATv2(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=4, num_layers=3, dropout=0.3):
        super(GATv2, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.edge_fc = Linear(edge_dim, hidden_dim)

        for i in range(num_layers):
            if i == 0:
                self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout))
            else:
                self.layers.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=hidden_dim, dropout=dropout))
            self.bn_layers.append(BatchNorm(hidden_dim * heads))

        self.pool = global_mean_pool
        self.fc1 = Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        edge_attr = F.relu(self.edge_fc(edge_attr))

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.bn_layers[i](x) 
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
