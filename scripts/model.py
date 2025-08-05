import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class Model(nn.Module):
    def __init__(self, h1, num_layers, input_size=2, output_size=10, freeze_layers=[]):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()  # List to hold layers
        self.input_size = input_size
        self.output_size = output_size

        # First layer
        self.layers.append(nn.Linear(self.input_size, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))  # Each hidden layer has h1 neurons

        # Output layer
        self.out = nn.Linear(h1, self.output_size)

        # Freeze selected layers
        for layer_idx in freeze_layers:
            if layer_idx < len(self.layers):  # Ensure valid layer index
                for param in self.layers[layer_idx].parameters():
                    param.requires_grad = False

    def forward(self, x):
        self.num_layers = 1
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  
        #x = self.out(x)  
        return x


class MixtureGNN(nn.Module):
    def __init__(self, graph_model, node_input_dim=11, hidden_dim=64, output_dim=10):
        super().__init__()

        if graph_model == "SAGEConv":
            graph = SAGEConv
        elif graph_model == "GATConv":
            graph = GATConv
        elif graph_model == "GCNConv":
            graph = GCNConv
        else:
            raise ValueError("Unknown GNN model.")

        self.conv1 = graph(node_input_dim, hidden_dim)
        self.conv2 = graph(hidden_dim, hidden_dim*2)
        self.conv3 = graph(hidden_dim*2, hidden_dim*4)
        self.conv4 = graph(hidden_dim*4, hidden_dim*2)
        self.conv5 = graph(hidden_dim*2, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))

        graph_embedding = global_mean_pool(x, batch)  

        return self.readout(graph_embedding)  



class MixtureEtchModel(nn.Module):
    def __init__(self, fnn_a, fnn_b, gnn, device):
        super().__init__()
        self.fnn_a = fnn_a
        self.fnn_b = fnn_b
        self.gnn = gnn
        self.device = device

        # Freeze pre-trained FNNs
        for param in self.fnn_a.parameters():
            param.requires_grad = False
        for param in self.fnn_b.parameters():
            param.requires_grad = False

    def forward(self, x):
        inputs = x[:, :2]             
        frac_a = x[:, 2]               

        out_a = self.fnn_a(inputs)     
        out_b = self.fnn_b(inputs)     

        '''node_a = (frac_a.unsqueeze(1) * out_a)        
                                 node_b = ((1 - frac_a).unsqueeze(1) * out_b)''' 

        node_a = torch.concatenate([out_a, frac_a.unsqueeze(1)], dim=1)        
        node_b = torch.concatenate([out_b, (1 - frac_a).unsqueeze(1)], dim=1)       

        # Create node features for all 2-node graphs
        node_feats = torch.stack([node_a, node_b], dim=1)  
        node_feats = node_feats.view(-1, node_feats.shape[-1])  

        # Build edge_index (2-node full graph) for each sample
        edge_index_base = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_index_list = []
        batch = []

        for i in range(x.shape[0]): 
            offset = i * 2
            edge_index_list.append(edge_index_base + offset)
            batch.extend([i, i])  

        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)  
        batch = torch.tensor(batch, dtype=torch.long).to(self.device)  

        return self.gnn(node_feats, edge_index, batch)
