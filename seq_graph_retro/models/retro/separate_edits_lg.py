import torch
from typing import List, Dict, Tuple, Union

from seq_graph_retro.models import SingleEdit, MultiEdit, LGClassifier, LGIndEmbed

EDIT_NET_DICT = {'SingleEdit': SingleEdit, "single_edit": SingleEdit, 'MultiEdit': MultiEdit}
LG_NET_DICT = {'LGClassifier': LGClassifier, "lg_classifier": LGClassifier,
        "lg_ind": LGIndEmbed, 'LGIndEmbed': LGIndEmbed, 'LGIndEmbedClassifier': LGIndEmbed}

class EditLGSeparate:

    def __init__(self,
                 edits_config: Dict,
                 lg_config: Dict,
                 edit_net_name: str = 'SingleEdit',
                 lg_net_name: str = 'LGClassifier',
                 device: str = 'cpu',
                 **kwargs):
        """
        Parameters
        ----------
        edits_config: Dict,
            Config for the edit prediction model
        lg_config: Dict,
            Config for the leaving group prediction model
        edit_net_name: str, default BondEdits,
            Name of the edit prediction network
        lg_net_name: str, default LGClassifier,
            Name of LGClassifier network
        """
        edit_model_class = EDIT_NET_DICT.get(edit_net_name)
        lg_model_class = LG_NET_DICT.get(lg_net_name)
        self.edit_net = edit_model_class(**edits_config, device=device)
        self.lg_net = lg_model_class(**lg_config, device=device)
        self.device = device

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True) if tensor is not None else None for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def _compute_edit_logits(self, graph_tensors: Union[Tuple[torch.Tensor], List[Tuple[torch.Tensor]]],
                             scopes: Tuple[List], ha=None, bg_inputs=None) -> Tuple[torch.Tensor]:
        """Computes the edit logits for given tensors.

        Parameters
        ----------
        graph_tensors: Union[Tuple[torch.Tensor], List[Tuple[torch.Tensor]]],
            Graph tensors used. Could be a List of these tensors, or just an
            individual one
        scopes: Tuple[List],

        """
        return self.edit_net._compute_edit_logits(graph_tensors, scopes, ha=ha, bg_inputs=bg_inputs)

    def _compute_lg_logits(self, graph_vecs_pad, prod_vecs, lg_labels=None):
        return self.lg_net._compute_lg_logits(graph_vecs_pad=graph_vecs_pad,
                                              prod_vecs=prod_vecs,
                                              lg_labels=lg_labels)

    def _compute_lg_step(self, graph_vecs, prod_vecs, prev_embed=None):
        return self.lg_net._compute_lg_step(graph_vecs=graph_vecs,
                                            prod_vecs=prod_vecs,
                                            prev_embed=prev_embed)

    def to(self, device: str) -> None:
        """Convert to device.

        Parameters
        ----------
        device: str,
            Device used
        """
        self.edit_net.to(device)
        self.lg_net.to(device)

    def eval(self) -> None:
        """Turn the network into eval mode."""
        self.edit_net.eval()
        self.lg_net.eval()

    def load_state_dict(self, edit_state: Dict, lg_state: Dict) -> None:
        """Loads state dict.

        Parameters
        ----------
        edit_state: Dict,
            State dict for the edit prediction network
        lg_state: Dict,
            State dict for the leaving group network.
        """
        self.edit_net.load_state_dict(edit_state)
        self.lg_net.load_state_dict(lg_state)

    def predict(self, prod_smi: str, rxn_class: int = None, **kwargs) -> Tuple[List]:
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        """
        edits = self.edit_net.predict(prod_smi, rxn_class=rxn_class)
        if not isinstance(edits, list):
            edits = [edits]

        try:
            labels = self.lg_net.predict(prod_smi, core_edits=edits, rxn_class=rxn_class)
            return edits, labels

        except:
            return edits, []
