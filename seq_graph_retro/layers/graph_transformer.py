import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from seq_graph_retro.layers.rnn import MPNLayer

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hsize: int, dropout_p: float = 0.15):
        """Initialization.
        :param size: the input dimension.
        :param dropout: the dropout ratio.
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hsize, elementwise_affine=True)
        self.dropout_layer = nn.Dropout(dropout_p)

    def forward(self, inputs, outputs):
        """Apply residual connection to any sublayer with the same size."""
        if inputs is None:
            return self.dropout_layer(self.norm(outputs))
        return inputs + self.dropout_layer(self.norm(outputs))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, in_dim, h_dim, out_dim=None, dropout_p=0.3, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.dropout_p = dropout_p
        self._build_components()

    def _build_components(self):
        self.W_1 = nn.Linear(self.in_dim, self.h_dim)
        self.W_2 = nn.Linear(self.h_dim, self.out_dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

    def forward(self, x):
        return self.W_2(self.dropout_layer(F.relu(self.W_1(x))))


class Head(nn.Module):

    def __init__(self,
                 rnn_type: str,
                 edge_fdim: int,
                 node_fdim: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs):
        super(Head, self).__init__(**kwargs)
        self.rnn_type = rnn_type
        self.edge_fdim = edge_fdim
        self.node_fdim = node_fdim
        self.hsize = hsize
        self.depth = depth
        self.dropout_p = dropout_p
        self._build_components()

    def _build_components(self):
        self.mpn_q = MPNLayer(rnn_type=self.rnn_type, edge_fdim=self.edge_fdim,
                                node_fdim=self.node_fdim, hsize=self.hsize,
                                depth=self.depth, dropout_p=self.dropout_p)
        self.mpn_k = MPNLayer(rnn_type=self.rnn_type, edge_fdim=self.edge_fdim,
                                node_fdim=self.node_fdim, hsize=self.hsize,
                                depth=self.depth, dropout_p=self.dropout_p)
        self.mpn_v = MPNLayer(rnn_type=self.rnn_type, edge_fdim=self.edge_fdim,
                                node_fdim=self.node_fdim, hsize=self.hsize,
                                depth=self.depth, dropout_p=self.dropout_p)

    def embed_graph(self, graph_tensors):
        """Replaces input graph tensors with corresponding feature vectors.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        """
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        hnode = fnode.clone()
        fmess1 = hnode.index_select(index=fmess[:, 0].long(), dim=0)
        fmess2 = fmess[:, 2:].clone()
        hmess = torch.cat([fmess1, fmess2], dim=-1)
        return hnode, hmess, agraph, bgraph

    def forward(self, graph_tensors, mask=None):
        graph_tensors = self.embed_graph(graph_tensors)
        q, _ = self.mpn_q(*graph_tensors, mask=mask)
        k, _ = self.mpn_k(*graph_tensors, mask=mask)
        v, _ = self.mpn_v(*graph_tensors, mask=mask)
        return q, k, v


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        # query: n_heads x n_atoms x dk
        # key: n_heads x n_atoms x dk
        # value: n_heads x n_atoms x dk
        scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).expand(scores.shape) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1) # n_heads x n_atoms x n_atoms
        if dropout is not None:
            p_attn = dropout(p_attn)
        scaled_vals = torch.bmm(p_attn, value).transpose(0, 1)
        return scaled_vals, p_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, hsize: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert hsize % n_heads == 0

        # We assume d_v always equals d_k
        self.hsize = hsize
        self.d_k = hsize // n_heads
        self.n_heads = n_heads  # number of heads
        self.bias = bias
        self.dropout = dropout
        self._build_components()

    def _build_components(self):
        self.linear_layers = nn.ModuleList([nn.Linear(self.hsize, self.hsize) for _ in range(3)])  # why 3: query, key, value
        self.output_linear = nn.Linear(self.hsize, self.hsize, self.bias)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, query, key, value, mask=None):
        n_atoms = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(n_atoms, self.n_heads, self.d_k).transpose(1, 0)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.contiguous().view(n_atoms, self.n_heads * self.d_k)
        return self.output_linear(x)


class MultiHeadBlock(nn.Module):

    def __init__(self,
                 rnn_type: str,
                 hsize: int,
                 depth: int,
                 n_heads: int,
                 node_fdim: int,
                 edge_fdim: int,
                 bias: bool = False,
                 dropout_p: float = 0.15,
                 res_connection: bool = False,
                 **kwargs):
        super(MultiHeadBlock, self).__init__(**kwargs)
        self.hsize = hsize
        self.rnn_type = rnn_type
        self.n_heads = n_heads
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.bias = bias
        self.dropout_p = dropout_p
        self.res_connection = res_connection
        self._build_layers()

    def _build_layers(self):
        self.W_i = nn.Linear(self.node_fdim, self.hsize, bias=False)
        self.W_o = nn.Linear(self.hsize, self.hsize, bias=self.bias)

        self.layernorm = nn.LayerNorm(self.hsize, elementwise_affine=True)
        self.heads = [Head(rnn_type=self.rnn_type, depth=self.depth,
                           hsize=self.hsize // self.n_heads, node_fdim=self.hsize,
                           edge_fdim=self.edge_fdim, dropout_p=self.dropout_p)
                      for _ in range(self.n_heads)]
        self.heads = nn.ModuleList(self.heads)
        self.attention = MultiHeadAttention(n_heads=self.n_heads, hsize=self.hsize,
                                            dropout=self.dropout_p, bias=self.bias)
        self.sub_layer = SublayerConnection(hsize=self.hsize, dropout_p=self.dropout_p)

    def forward(self, graph_tensors, scopes):
        fnode, fmess, agraph, mess_graph, _ = graph_tensors
        queries, keys, values = [], [], []

        if fnode.size(1) != self.hsize:
            fnode = self.W_i(fnode)

        tensors = (fnode,) + tuple(graph_tensors[1:])
        for head in self.heads:
            q, k, v = head(tensors)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))

        n_atoms = q.size(0)
        dk, dv = q.size(1), v.size(1)
        queries = torch.cat(queries, dim=1).view(n_atoms, -1) # n_atoms x hsize
        keys = torch.cat(keys, dim=0).view(n_atoms, -1) # n_atoms x hsize
        values = torch.cat(values, dim=0).view(n_atoms, -1) # n_atoms x hsize

        assert queries.shape == (n_atoms, self.hsize)
        assert keys.shape == (n_atoms, self.hsize)
        assert values.shape == (n_atoms, self.hsize)

        # This boolean mask is for making sure attention only happens over
        # atoms of the same molecule
        mask = queries.new_zeros(n_atoms, n_atoms)
        a_scope = scopes[0]
        for a_start, a_len in a_scope:
            mask[a_start: a_start + a_len, a_start: a_start + a_len] = 1
        mask[0, 0] = 1

        x_out = self.attention(queries, keys, values, mask=mask)
        x_out = self.W_o(x_out)

        x_in = None
        if self.res_connection:
            x_in = fnode

        h_atom = self.sub_layer(x_in, x_out)
        next_tensors = (h_atom,) + graph_tensors[1:]
        return next_tensors, scopes
