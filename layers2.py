import torch.nn as nn
import torch
from torch.nn import functional as F


class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer1, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # self.linear_W = nn.Linear(in_features, out_features)
        # self.linear_a1 = nn.Linear(out_features, 1)
        # self.linear_a2 = nn.Linear(out_features, 1)
        self.W_transform = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(2*out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_paramemters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, h, adj):
        Wh = self.W_transform(h)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + torch.transpose(Wh2, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# read out层的定义
class Readout(nn.Module):
    def __init__(self, input_dim, n_class, dropout=0.5, act=torch.relu, bias=False):
        super(Readout, self).__init__()
        self.input_dim = input_dim
        self.output_dim = n_class
        self.act = act
        self.bias = bias
        self.emb = nn.Linear(input_dim, input_dim, bias=True)
        self.att = nn.Linear(input_dim, 1, bias=True)
        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_dim, n_class, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        att = self.att(x).sigmoid()
        emb = self.act(self.emb(x))
        x = att * emb
        x = self._max(x, mask) + self._mean(x, mask)
        x = self.mlp(x)

        return x

    def _max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def _mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class GAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, dropout, alpha):
        """Dense version of GAT."""
        super(GAT1, self).__init__()
        self.dropout = dropout
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        self.attentions = [GraphAttentionLayer1(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer1(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.read = Readout(nhid, nclass, dropout=dropout)

    def forward(self, x, adj, mask):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        output = self.read(x, mask)
        return F.log_softmax(output, dim=1)