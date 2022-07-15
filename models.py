import torch_geometric.utils
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
from torch_cluster import random_walk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class RandomWalkingRewirer(torch.nn.Module):
    def __init__(self, num_steps, keep_steps):
        super(RandomWalkingRewirer, self).__init__()

        self.num_steps = num_steps
        self.keep_steps = keep_steps

    def forward(self, edge_index, num_nodes):
        ''' Returns a new edge_index for each path step based on policy walking'''
        row, col = edge_index

        walk_paths = random_walk(row, col, start=torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device), walk_length=self.num_steps, p= 1, q = 1, coalesced= True)

        added_edges = []
        for i in range(1, self.num_steps):
            if i+1 in self.keep_steps:
                added_edges.append(walk_paths[:, [0, i + 1]].T)

        return added_edges

class RWGAT(torch.nn.Module):
    """Rewired Walker GAT"""

    def __init__(self, policy, num_node_features, dim_h, heads, num_classes, num_steps):
        super(RWGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, dim_h, heads)
        self.conv2 = GATConv(dim_h * heads, dim_h, heads)
        self.conv3 = GATConv(dim_h*heads, num_classes, 1)


        if policy == 'random_walk':
            self.rewirer = RandomWalkingRewirer(num_steps = num_steps, keep_steps=[int(num_steps/2), num_steps])
        elif policy == 'none':
            self.rewirer = None
        else:
            raise NotImplementedError("Current allowable policies are 'random_walk' or 'none'")

    def rewired_edge_index(self, edge_index, added_edges):
        if added_edges is None:
            return edge_index
        else:
            new_edge_index = torch.cat((edge_index, added_edges), dim=1)
            new_edge_index = torch_geometric.utils.to_undirected(new_edge_index)
            return new_edge_index

    def forward(self, x, edge_index):
        # first compute edges to be added
        if self.rewirer is not None:
            added_edges = self.rewirer(edge_index, x.shape[0])
        else:
            added_edges = [None, None]

        # node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, self.rewired_edge_index(edge_index, added_edges[0]))
        h3 = self.conv2(h2, self.rewired_edge_index(edge_index, added_edges[1]))

        return F.log_softmax(h3, dim=1)

    def compute_loss(self, model_output, labels):
        # compute the negative log likelihood loss
        return F.nll_loss(model_output, labels)


if __name__ == '__main__':
    sizes = [14, 35]  # list of sizes for  the individual blocks
    probs = [[0.7, 0.05], [0.05, 0.6]]  # probabilites of edges connecting different blocks
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    pos = nx.spring_layout(g)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(g, pos, alpha=0.2)
    nx.draw_networkx_nodes(g, pos, node_color='k', node_size=20, linewidths=6)
    # plt.show()

    data = torch_geometric.utils.from_networkx(g)

    data.x = torch.tensor(np.array(list(pos.values())), dtype=torch.float)

    rwgat = RWGAT(policy='random_walk', num_node_features=2, dim_h=8, heads=8, num_classes=2, num_steps=4)
    print(rwgat(data.x, data.edge_index))
