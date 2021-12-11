import torch
from rdkit import Chem
from typing import Tuple, List

from seq_graph_retro.data.collate_fns import pack_graph_feats
from seq_graph_retro.data import BaseDataset, EvalDataset
from seq_graph_retro.utils.parse import ReactionInfo
from seq_graph_retro.molgraph import MultiElement, RxnElement

def prep_graphs(info: ReactionInfo) -> Tuple[RxnElement, MultiElement]:
    """Prepares reaction graphs for the collater.

    Parameters
    ----------
    info: ReactionInfo,
        ReactionInfo for the particular reaction.
    """
    r, p = info.rxn_smi.split(">>")
    return (RxnElement(Chem.MolFromSmiles(p), rxn_class=info.rxn_class),
    MultiElement(Chem.MolFromSmiles(r), rxn_class=info.rxn_class))

def prep_dgi_eval(info: ReactionInfo) -> List[RxnElement]:
    """Prepares a list of RxnElements for the DGI collater.

    Parameters
    ----------
    info: ReactionInfo,
        ReactionInfo for a particular reaction.
    """
    rxn_elements = []
    r, p = info.rxn_smi.split(">>")
    rxn_elements.append(RxnElement(Chem.MolFromSmiles(p)))
    rxn_elements.extend([RxnElement(Chem.MolFromSmiles(smi)) for smi in r.split(".")])
    return rxn_elements

class EncoderDataset(BaseDataset):

    def collater(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        prod_inputs, reac_inputs, frag_inputs = attributes
        return prod_inputs, reac_inputs, frag_inputs

class EncoderEvalDataset(EvalDataset):

    def collater(self, attributes: List[ReactionInfo]) -> Tuple[torch.Tensor]:
        rxn_smi_batch = [prep_graphs(info) for info in attributes]
        prod_batch, reac_batch = list(zip(*rxn_smi_batch))
        prod_inputs = pack_graph_feats(prod_batch, directed=True, use_rxn_class=self.use_rxn_class)
        reac_inputs = pack_graph_feats(reac_batch, directed=True, use_rxn_class=self.use_rxn_class)
        return prod_inputs, reac_inputs

class ContextPredDataset(BaseDataset):

    def collater(self, attributes: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        substruct_inputs, context_inputs, root_idxs, overlaps, overlap_scopes = attributes
        return substruct_inputs, context_inputs, root_idxs, overlaps, overlap_scopes
