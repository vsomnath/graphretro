import torch
import torch.nn as nn
from typing import Tuple

from seq_graph_retro.utils.torch import index_select_ND, index_scatter

class MPNLayer(nn.Module):
    """MessagePassing Network based encoder. Messages are updated using an RNN
    and the final message is used to update atom embeddings."""

    def __init__(self,
                 rnn_type: str,
                 node_fdim: int,
                 edge_fdim: int,
                 hsize: int,
                 depth: int,
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
        super(MPNLayer, self).__init__(**kwargs)
        self.hsize = hsize
        self.edge_fdim = edge_fdim
        self.rnn_type = rnn_type
        self.depth = depth
        self.node_fdim = node_fdim
        self.dropout_p = dropout_p
        self._build_layers()

    def _build_layers(self) -> None:
        """Build layers associated with the MPNLayer."""
        self.W_o = nn.Sequential(nn.Linear(self.node_fdim + self.hsize, self.hsize), nn.ReLU())
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

    def forward(self, fnode: torch.Tensor, fmess: torch.Tensor,
                      agraph: torch.Tensor, bgraph: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor]:
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
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0 #first node is padding

        return node_hiddens * mask, h


class GRU(nn.Module):
    """GRU Message Passing layer."""

    def __init__(self,
                 input_size: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        input_size: int,
            Size of the input
        hsize: int,
            Hidden state size
        depth: int,
            Number of time steps of message passing
        device: str, default cpu
            Device used for training
        """
        super(GRU, self).__init__(**kwargs)
        self.hsize = hsize
        self.input_size = input_size
        self.depth = depth
        self.dropout_p = dropout_p
        self._build_layer_components()

    def _build_layer_components(self) -> None:
        """Build layer components."""
        self.W_z = nn.Linear(self.input_size + self.hsize, self.hsize)
        self.W_r = nn.Linear(self.input_size, self.hsize, bias=False)
        self.U_r = nn.Linear(self.hsize, self.hsize)
        self.W_h = nn.Linear(self.input_size + self.hsize, self.hsize)

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)

    def get_init_state(self, fmess: torch.Tensor, init_state: torch.Tensor = None) -> torch.Tensor:
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        return h if init_state is None else torch.cat( (h, init_state), dim=0)

    def get_hidden_state(self, h: torch.Tensor) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state of the GRU
        """
        return h

    def GRU(self, x: torch.Tensor, h_nei: torch.Tensor) -> torch.Tensor:
        """Implements the GRU gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        """
        sum_h = h_nei.sum(dim=1)
        z_input = torch.cat([x,sum_h], dim=1)
        z = torch.sigmoid(self.W_z(z_input))

        r_1 = self.W_r(x).view(-1, 1, self.hsize)
        r_2 = self.U_r(h_nei)
        r = torch.sigmoid(r_1 + r_2)

        gated_h = r * h_nei
        sum_gated_h = gated_h.sum(dim=1)
        h_input = torch.cat([x,sum_gated_h], dim=1)
        pre_h = torch.tanh(self.W_h(h_input))
        new_h = (1.0 - z) * sum_h + z * pre_h
        return new_h

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0 #first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            h = self.GRU(fmess, h_nei)
            h = h * mask
            h = self.dropouts[i](h)
        return h

    def sparse_forward(self, h: torch.Tensor, fmess: torch.Tensor,
                       submess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Unknown use.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state tensor
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        submess: torch.Tensor,
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            sub_h = self.GRU(fmess, h_nei)
            h = index_scatter(sub_h, h, submess)
        return h

class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hsize: int,
                 depth: int,
                 dropout_p: float = 0.15,
                 **kwargs):
        """
        Parameters
        ----------
        input_size: int,
            Size of the input
        hsize: int,
            Hidden state size
        depth: int,
            Number of time steps of message passing
        device: str, default cpu
            Device used for training
        """
        super(LSTM, self).__init__(**kwargs)
        self.hsize = hsize
        self.input_size = input_size
        self.depth = depth
        self.dropout_p = dropout_p
        self._build_layer_components()

    def _build_layer_components(self):
        """Build layer components."""
        self.W_i = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_o = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W_f = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Sigmoid())
        self.W = nn.Sequential(nn.Linear(self.input_size + self.hsize, self.hsize), nn.Tanh())

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)

    def get_init_state(self, fmess: torch.Tensor,
                       init_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the initial hidden state of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        init_state: torch.Tensor, default None
            Custom initial state supplied.
        """
        h = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        c = torch.zeros(len(fmess), self.hsize, device=fmess.device)
        if init_state is not None:
            h = torch.cat((h, init_state), dim=0)
            c = torch.cat((c, torch.zeros_like(init_state)), dim=0)
        return h,c

    def get_hidden_state(self, h: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Gets the hidden state.

        Parameters
        ----------
        h: Tuple[torch.Tensor, torch.Tensor],
            Hidden state tuple of the LSTM
        """
        return h[0]

    def LSTM(self, x: torch.Tensor, h_nei: torch.Tensor, c_nei: torch.Tensor) -> torch.Tensor:
        """Implements the LSTM gating equations.

        Parameters
        ----------
        x: torch.Tensor,
            Input tensor
        h_nei: torch.Tensor,
            Hidden states of the neighbors
        c_nei: torch.Tensor,
            Memory state of the neighbors
        """
        h_sum_nei = h_nei.sum(dim=1)
        x_expand = x.unsqueeze(1).expand(-1, h_nei.size(1), -1)
        i = self.W_i( torch.cat([x, h_sum_nei], dim=-1) )
        o = self.W_o( torch.cat([x, h_sum_nei], dim=-1) )
        f = self.W_f( torch.cat([x_expand, h_nei], dim=-1) )
        u = self.W( torch.cat([x, h_sum_nei], dim=-1) )
        c = i * u + (f * c_nei).sum(dim=1)
        h = o * torch.tanh(c)
        return h, c

    def forward(self, fmess: torch.Tensor, bgraph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RNN.

        Parameters
        ----------
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        c = torch.zeros(fmess.size(0), self.hsize, device=fmess.device)
        mask = torch.ones(h.size(0), 1, device=h.device)
        mask[0, 0] = 0 #first message is padding

        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            h,c = self.LSTM(fmess, h_nei, c_nei)
            h = h * mask
            c = c * mask
            h = self.dropouts[i](h)
            c = self.dropouts[i](c)
        return h,c

    def sparse_forward(self, h: torch.Tensor, fmess: torch.Tensor,
                       submess: torch.Tensor, bgraph: torch.Tensor) -> torch.Tensor:
        """Unknown use.

        Parameters
        ----------
        h: torch.Tensor,
            Hidden state tensor
        fmess: torch.Tensor,
            Contains the initial features passed as messages
        submess: torch.Tensor,
        bgraph: torch.Tensor,
            Bond graph tensor. Contains who passes messages to whom.
        """
        h,c = h
        mask = h.new_ones(h.size(0)).scatter_(0, submess, 0)
        h = h * mask.unsqueeze(1)
        c = c * mask.unsqueeze(1)
        for i in range(self.depth):
            h_nei = index_select_ND(h, 0, bgraph)
            c_nei = index_select_ND(c, 0, bgraph)
            sub_h, sub_c = self.LSTM(fmess, h_nei, c_nei)
            h = index_scatter(sub_h, h, submess)
            c = index_scatter(sub_c, c, submess)
        return h,c
