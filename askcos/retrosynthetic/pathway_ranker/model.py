"""
This module defines the architectures for TreeLSTM and PathwayRankingModel.
"""

import torch


class TreeLSTM(torch.nn.Module):
    """
    PyTorch TreeLSTM model that implements efficient batching.
    """

    def __init__(self, in_features, out_features):
        """
        TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained separate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order):
        """
        Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        """

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return h, c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        """
        Helper function to evaluate all tree nodes currently able to be evaluated.
        """
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]
        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            # c = c.clone()
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])


class PathwayRankingModel(torch.nn.Module):
    def __init__(self, fp_size, lstm_size, encoder=False):
        super().__init__()
        # prd fp encoding
        self.encoder = encoder
        self.pfp_layer = torch.nn.Sequential(
            torch.nn.Linear(fp_size, lstm_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(lstm_size, lstm_size),
            torch.nn.ReLU())
        # rxn fp encoding
        self.rxnfp_layer = torch.nn.Sequential(
            torch.nn.Linear(fp_size, lstm_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(lstm_size, lstm_size),
            torch.nn.ReLU())
        # tree lstm
        self.tree_lstm = TreeLSTM(lstm_size, lstm_size)

        # final scorer
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(lstm_size, int(lstm_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(int(lstm_size / 2), 1))

    def forward(self, pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes):
        pfp_out = self.pfp_layer(pfp)
        rxnfp_out = self.rxnfp_layer(rxnfp)
        # multiply elemental wise
        lstm_input = torch.tanh(pfp_out * rxnfp_out)
        # go through tree lstm
        lstm_output, _ = self.tree_lstm(lstm_input, node_order, adjacency_list, edge_order)

        # calculate tree score, the input should be the top node
        scorer_input = torch.stack([t[0, :] for t in torch.split(lstm_output, num_nodes, dim=0)])

        score = self.scorer(scorer_input)

        if not self.encoder:
            return score
        else:
            return score, scorer_input
