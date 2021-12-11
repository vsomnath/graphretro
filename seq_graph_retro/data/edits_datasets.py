import torch
from typing import Tuple, List, Optional

from seq_graph_retro.utils.parse import ReactionInfo
from seq_graph_retro.data import BaseDataset, EvalDataset

class SingleEditDataset(BaseDataset):

    def collater(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        prod_inputs, edit_labels, frag_inputs, lg_labels, lengths, bg_inputs = attributes
        prod_tensors, prod_scopes = prod_inputs
        return prod_tensors, prod_scopes, bg_inputs, edit_labels

class MultiEditDataset(BaseDataset):

    def collater(self, attributes: List[Tuple[torch.tensor]]) -> Tuple[torch.Tensor]:
        assert isinstance(attributes, list)
        assert len(attributes) == 1

        attributes = attributes[0]
        prod_seq_inputs, edit_labels, seq_masks, frag_inputs, lg_labels, lengths = attributes
        return prod_seq_inputs, edit_labels, seq_masks


class EditsEvalDataset(EvalDataset):

    def collater(self, attributes: List[ReactionInfo]) -> Tuple[str, List[str], Optional[List[int]]]:
        info_batch = attributes
        prod_smi = [info.rxn_smi.split(">>")[-1] for info in info_batch]
        core_edits = [set(info.core_edits) for info in info_batch]

        if self.use_rxn_class:
            rxn_classes = [info.rxn_class for info in info_batch]
            return prod_smi, core_edits, rxn_classes
        else:
            return prod_smi, core_edits, None
