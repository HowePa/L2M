import torch
import torch.nn as nn
import torch.nn.functional as F
from EEN.message_pass import update_efeat


class GraphConv(nn.Module):

    def __init__(self, in_feats, out_feats, activation=None):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2 * in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, feat, graph, mask=None, norm=None):
        feat = mask.double().unsqueeze(-1) * feat
        rst = update_efeat(graph, feat * norm) * norm
        # update feature
        rst = torch.cat([rst, feat], dim=-1)
        # rst = (rst + feat) * norm
        rst = torch.matmul(rst, self.weight.double())
        # bias
        rst = rst + self.bias
        if self._activation is not None:
            rst = self._activation(rst)
        return rst


class PolicyGraphConvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PolicyGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, hidden_dim, activation=F.relu))

        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))

        self.layers.append(GraphConv(hidden_dim, output_dim, activation=None))

    def forward(self, h, g, mask=None):
        masked_deg = update_efeat(g, mask)
        masked_deg.masked_fill_(masked_deg == 0, 1)
        norm = torch.pow(masked_deg, -0.5).unsqueeze(-1)

        for _, layer in enumerate(self.layers):
            h = layer(h, g, mask=mask, norm=norm)
        return h


class ValueGraphConvNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ValueGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, hidden_dim, activation=F.relu))

        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))

        self.layers.append(GraphConv(hidden_dim, output_dim, activation=None))

    def forward(self, h, g, mask=None):
        masked_deg = update_efeat(g, mask)
        masked_deg.masked_fill_(masked_deg == 0, 1)
        norm = torch.pow(masked_deg, -0.5).unsqueeze(-1)

        for _, layer in enumerate(self.layers):
            h = layer(h, g, mask=mask, norm=norm)
        return h
