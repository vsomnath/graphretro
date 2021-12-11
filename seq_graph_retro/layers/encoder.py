import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from seq_graph_retro.molgraph.mol_features import ATOM_FDIM, BOND_FDIM
from seq_graph_retro.layers.rnn import GRU, LSTM, MPNLayer
from seq_graph_retro.layers.graph_transformer import MultiHeadBlock, PositionwiseFeedForward, SublayerConnection
from seq_graph_retro.utils.torch import index_select_ND

Scope = List[Tuple[int, int]]


class LogitEncoder(nn.Module):

    def __init__(self,
                 rnn_type: str,
                 node_fdim: int,
                 edge_fdim: int,
                 hsize: int,
                 depth: int,
                 outdim: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        rnn_type: str,
            Type of RNN used (gru/lstm)
        input_size: int,
            Input size
        node_fdim: int,
            Number of node features
        hsize: int,
            Hidden state size
        depth: int,
            Number of timesteps in the RNN
        """
        super(LogitEncoder, self).__init__(**kwargs)
        self.hsize = hsize
        self.edge_fdim = edge_fdim
        self.rnn_type = rnn_type
        self.depth = depth
        self.node_fdim = node_fdim
        self.outdim = outdim
        self.dropout_p = dropout_p
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNLayer."""
        if self.rnn_type == 'gru':
            self.rnn = GRU(input_size=self.node_fdim + self.edge_fdim,
                           hsize=self.hsize,
                           depth=self.depth,
                           dropout_p=self.dropout_p)
        elif self.rnn_type == 'lstm':
            self.rnn = LSTM(input_size=self.node_fdim + self.edge_fdim,
                           hsize=self.hsize,
                           depth=self.depth,
                           dropout_p=self.dropout_p)
        else:
            raise ValueError('unsupported rnn cell type ' + self.rnn_type)

        add_dim = self.hsize
        self.W_f = nn.Sequential(nn.Linear(self.node_fdim + add_dim, self.outdim), nn.Sigmoid())
        self.W_i = nn.Sequential(nn.Linear(self.node_fdim + add_dim, self.outdim), nn.Sigmoid())
        self.W_m = nn.Sequential(nn.Linear(self.node_fdim + add_dim, self.hsize), nn.ReLU(),
                                 nn.Linear(self.hsize, self.outdim))

    def embed_graph(self, graph_tensors: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
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

    def forward(self, logits: torch.Tensor, graph_tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MPNLayer.

        Parameters
        ----------
        fnode: torch.Tensor,
            Node feature tensor
        fmess: torch.Tensor,
            Message features
        agraph: torch.Tensor,
            Neighborhood of an atom
        bgraph: torch.Tensor,
            Neighborhood of a bond, except the directed bond from the destination
            node to the source node
        mask: torch.Tensor,
            Masks on nodes
        """
        m = logits
        fnode, fmess, agraph, bgraph = self.embed_graph(graph_tensors)
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)

        f = self.W_f(torch.cat([fnode, nei_message], dim=1))
        i = self.W_i(torch.cat([fnode, nei_message], dim=1))
        mtilde = self.W_m(torch.cat([fnode, nei_message], dim=1))
        m = f * m + i * mtilde

        if mask is None:
            mask = torch.ones(m.size(0), 1, device=fnode.device)
            mask[0, 0] = 0 #first node is padding

        return m * mask


class GraphFeatEncoder(nn.Module):
    """
    GraphFeatEncoder encodes molecules by using features of atoms and bonds,
    instead of a vocabulary, which is used for generation tasks.
    """

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 rnn_type: str,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        node_fdim: int,
            Number of atom features
        edge_fdim: int,
            Number of bond features
        rnn_type: str,
            Type of RNN used for encoding
        hsize: int,
            Hidden state size
        depth: int,
            Number of timesteps in the RNN
        """
        super(GraphFeatEncoder, self).__init__(**kwargs)
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.rnn_type = rnn_type
        self.atom_size = node_fdim
        self.hsize = hsize
        self.depth = depth
        self.dropout_p = dropout_p

        self._build_layers()

    def _build_layers(self):
        """Build layers associated with the GraphFeatEncoder."""
        self.encoder = MPNLayer(rnn_type=self.rnn_type,
                                  edge_fdim=self.edge_fdim,
                                  node_fdim=self.node_fdim,
                                  hsize=self.hsize, depth=self.depth,
                                  dropout_p=self.dropout_p)

    def embed_graph(self, graph_tensors: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
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

    def forward(self, graph_tensors: Tuple[torch.Tensor], scopes: Scope) -> Tuple[torch.Tensor]:
        """
        Forward pass of the graph encoder. First the feature vectors are extracted,
        and then encoded.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tuple of graph tensors - Contains atom features, message vector details,
            atom graph and bond graph for encoding neighborhood connectivity.
        scopes: Tuple[List]
            Scopes is composed of atom and bond scopes, which keep track of
            atom and bond indices for each molecule in the 2D feature list
        """
        tensors = self.embed_graph(graph_tensors)
        hatom,_ = self.encoder(*tensors, mask=None)
        atom_scope, bond_scope = scopes

        if isinstance(atom_scope[0], list):
            hmol = [torch.stack([hatom[st: st + le].sum(dim=0) for (st, le) in scope])
                    for scope in atom_scope]
        else:
            hmol = torch.stack([hatom[st: st + le].sum(dim=0) for st, le in atom_scope])
        return hmol, hatom


class WLNEncoder(nn.Module):
    """
    WLNEncoder encodes molecules by using features of atoms and bonds, following
    the update rules as defined in the WLN Architecture.
    """
    def __init__(self,
                 n_atom_feat: int = ATOM_FDIM,
                 n_bond_feat: int = BOND_FDIM,
                 hsize: int = 64,
                 depth: int = 3,
                 bias: bool = False,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        n_atom_feat: int, default ATOM_FDIM(87)
            Number of atom features
        n_bond_feat: int, default BOND_FDIM(6)
            Number of bond features
        hsize: int, default 64
            Size of the embeddings
        depth: int, default 3
            Depth of the WLN Graph Convolution
        bias: bool, default False
            Whether to use a bias term in the linear layers
        """
        super(WLNEncoder, self).__init__(**kwargs)
        self.n_atom_feat = n_atom_feat
        self.n_bond_feat = n_bond_feat
        self.hsize = hsize
        self.depth = depth
        self.bias = bias
        self.dropout_p = dropout_p
        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated."""
        self.atom_emb = nn.Linear(self.n_atom_feat, self.hsize, self.bias)
        self.bond_emb = nn.Linear(self.n_bond_feat, self.hsize, self.bias)

        self.U1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.U2 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.V = nn.Linear(2 * self.hsize, self.hsize, self.bias)

        self.W0 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W2 = nn.Linear(self.hsize, self.hsize, self.bias)

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, inputs: Tuple[torch.Tensor], scopes: Scope,
                return_layers: bool = False) -> Tuple[torch.Tensor]:
        """Forward propagation step.

        Parameters
        ----------
        inputs: tuple of torch.tensors
            Graph tensors used as input for the WLNEmbedding
        scopes: Tuple[List]
            Scopes is composed of atom and bond scopes, which keep track of
            atom and bond indices for each molecule in the 2D feature list
        return layers: bool, default False,
            Whether to return the atom embeddings for every layer of graph convolutions
        """
        layers = []
        atom_feat, bond_feat, atom_graph, bond_graph, _ = inputs
        atom_scope, bond_scope = scopes
        bs = len(atom_scope)

        h_atom = self.atom_emb(atom_feat)
        h_bond = self.bond_emb(bond_feat)

        for i in range(self.depth):
            h_atom_nei = index_select_ND(h_atom, dim=0, index=atom_graph)
            h_bond_nei = index_select_ND(h_bond, dim=0, index=bond_graph)

            f_atom = self.W0(h_atom)
            f_bond_nei = self.W1(h_bond_nei)
            f_atom_nei = self.W2(h_atom_nei)

            c_atom = f_atom * torch.sum(f_atom_nei * f_bond_nei, dim=1)
            layers.append(c_atom)

            nei_label = F.relu(self.V(torch.cat([h_bond_nei, h_atom_nei], dim=-1)))
            nei_label = torch.sum(nei_label, dim=1)
            new_label = self.U1(h_atom) + self.U2(nei_label)
            h_atom = F.relu(new_label)
            h_atom = self.dropouts[i](h_atom)

        c_atom_final = layers[-1]
        if isinstance(atom_scope[0], list):
            c_mol = [torch.stack([c_atom_final[st: st + le].sum(dim=0)
                     for (st, le) in scope]) for scope in atom_scope]
            assert len(c_mol) == bs
            assert c_mol[0].shape[-1] == self.hsize
        else:
            c_mol = torch.stack([c_atom_final[st: st + le].sum(dim=0)
                                 for st, le in atom_scope])
            assert len(c_mol) == bs
            assert c_mol.shape[-1] == self.hsize

        if return_layers:
            return c_mol, layers

        return c_mol, layers[-1]

class GTransEncoder(nn.Module):

    def __init__(self,
                 rnn_type: str,
                 hsize: int,
                 depth: int,
                 n_heads: int,
                 node_fdim: int,
                 edge_fdim: int,
                 n_mt_blocks: int,
                 bias: bool = False,
                 dropout_p: float = 0.15,
                 res_connection: bool = False,
                 **kwargs):
        super(GTransEncoder, self).__init__(**kwargs)
        self.hsize = hsize
        self.rnn_type = rnn_type
        self.n_heads = n_heads
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.bias = bias
        self.dropout_p = dropout_p
        self.res_connection = res_connection
        self.n_mt_blocks = n_mt_blocks
        self._build_layers()

    def _build_layers(self):
        self.mt_blocks = []
        node_fdim = self.node_fdim
        for i in range(self.n_mt_blocks):
            if i != 0:
                node_fdim = self.hsize
            self.mt_blocks.append(MultiHeadBlock(rnn_type=self.rnn_type, hsize=self.hsize, depth=self.depth,
                                                 n_heads=self.n_heads, node_fdim=node_fdim,
                                                 edge_fdim=self.edge_fdim, dropout_p=self.dropout_p,
                                                 res_connection=self.res_connection))
        self.mt_blocks = nn.ModuleList(self.mt_blocks)
        self.positionwise_mlp = PositionwiseFeedForward(in_dim=self.hsize + self.node_fdim,
                                                        h_dim=self.hsize * 2,
                                                        out_dim=self.hsize,
                                                        dropout_p=self.dropout_p)
        self.atom_sublayer = SublayerConnection(hsize=self.hsize, dropout_p=self.dropout_p)

    def update_atom_embeddings(self, hatom: torch.Tensor, fnode: torch.Tensor) -> torch.Tensor:
        hatom = torch.cat([hatom, fnode], dim=1)
        hatom = self.positionwise_mlp(hatom)
        return self.atom_sublayer(None, hatom)

    def forward(self, graph_tensors: Tuple[torch.Tensor], scopes: Scope) -> Tuple[torch.Tensor]:
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        tensors = (fnode, fmess, agraph, bgraph, _)

        for block in self.mt_blocks:
            tensors, scopes = block(tensors, scopes)
        hatom = tensors[0]
        atom_scope, bond_scope = scopes
        hatom = self.update_atom_embeddings(hatom, fnode)

        if isinstance(atom_scope[0], list):
            hmol = [torch.stack([hatom[st: st + le].sum(dim=0) for (st, le) in scope])
                    for scope in atom_scope]
        else:
            hmol = torch.stack([hatom[st: st + le].sum(dim=0) for st, le in atom_scope])
        return hmol, hatom
