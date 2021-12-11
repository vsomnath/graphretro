import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union
from rdkit import Chem

from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.utils.torch import build_mlp
from seq_graph_retro.utils.metrics import get_accuracy_lg
from seq_graph_retro.layers import AtomAttention, GraphFeatEncoder, WLNEncoder

from seq_graph_retro.utils.parse import apply_edits_to_mol
from seq_graph_retro.data.collate_fns import pack_graph_feats
from seq_graph_retro.molgraph.rxn_graphs import MultiElement, RxnElement


class LGIndEmbed(nn.Module):
    """LGIndEmbed is a classifier for predicting leaving groups on fragments."""

    def __init__(self,
                 config: Dict,
                 lg_vocab: Vocab,
                 encoder_name: str,
                 toggles: Dict = None,
                 device: str = 'cpu',
                 **kwargs):
        """
        Parameters
        ----------
        config: Dict,
            Config for all sub-modules and self
        lg_vocab: Vocab
            Vocabulary of leaving groups
        encoder_name: str,
            Name of the encoder network
        use_prev_pred: bool, default True
            Whether to use previous leaving group prediction
        device: str
            Device on which program runs
        """
        super(LGIndEmbed, self).__init__(**kwargs)
        self.config = config
        self.lg_vocab = lg_vocab
        self.encoder_name = encoder_name
        self.toggles = toggles if toggles is not None else {}
        self.device = device
        self.E_lg = torch.eye(len(lg_vocab)).to(device)

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the layers in the classifier."""
        config = self.config
        if self.encoder_name == 'GraphFeatEncoder':
            self.encoder = GraphFeatEncoder(node_fdim=config['n_atom_feat'],
                                            edge_fdim=config['n_bond_feat'],
                                            rnn_type=config['rnn_type'],
                                            hsize=config['mpn_size'],
                                            depth=config['depth'],
                                            dropout_p=config['dropout_mpn'])

        elif self.encoder_name == 'WLNEncoder':
            self.encoder = WLNEncoder(node_fdim=config['n_atom_feat'],
                                      edge_fdim=config['n_bond_feat'],
                                      hsize=config['mpn_size'],
                                      depth=config['depth'],
                                      bias=config['bias'],
                                      dropout_p=config['dropout_mpn'])
        else:
            raise ValueError()

        if self.toggles.get('use_attn', False):
            self.attn_layer = AtomAttention(n_bin_feat=config['n_bin_feat'],
                                            hsize=config['mpn_size'],
                                            n_heads=config['n_heads'],
                                            bias=config['bias'])

        lg_score_in_dim = 2 * config['mpn_size']
        if self.toggles.get('use_prev_pred', False):
            lg_score_in_dim += config['embed_size']

        self.lg_embedding = nn.Linear(in_features=len(self.lg_vocab),
                                      out_features=config['embed_size'],
                                      bias=config['embed_bias'])

        self.lg_score = build_mlp(in_dim=lg_score_in_dim,
                                  h_dim=config['mlp_size'],
                                  out_dim=len(self.lg_vocab),
                                  dropout_p=config['dropout_mlp'])

        self.lg_loss = nn.CrossEntropyLoss(ignore_index=self.lg_vocab["<pad>"])

    def _compute_lg_step(self, graph_vecs, prod_vecs, prev_embed=None):
        if self.toggles.get('use_prev_pred', False):
            if prev_embed is None:
                init_state = torch.zeros(graph_vecs.size(0), len(self.lg_vocab), device=self.device)
                init_state[:, self.lg_vocab.get("<bos>")] = 1
                prev_lg_emb = self.lg_embedding(init_state)
            else:
                prev_lg_emb = prev_embed

        if self.toggles.get('use_prev_pred', False):
            scores_lg = self.lg_score(torch.cat([prev_lg_emb, prod_vecs, graph_vecs], dim=-1))
        else:
            scores_lg = self.lg_score(torch.cat([prod_vecs, graph_vecs], dim=-1))
        return scores_lg, None

    def _compute_lg_logits(self, graph_vecs_pad, prod_vecs, lg_labels=None):
        scores = torch.tensor([], device=self.device)
        prev_lg_emb = None

        if lg_labels is None:
            for idx in range(graph_vecs_pad.size(1)):
                scores_lg, _ = self._compute_lg_step(graph_vecs_pad[:, idx], prod_vecs, prev_embed=prev_lg_emb)
                prev_lg_emb = self.lg_embedding(self.E_lg.index_select(index=torch.argmax(scores_lg, dim=-1), dim=0))
                scores = torch.cat([scores, scores_lg.unsqueeze(1)], dim=1)

        else:
            for idx in range(graph_vecs_pad.size(1)):
                scores_lg, _ = self._compute_lg_step(graph_vecs_pad[:, idx], prod_vecs, prev_embed=prev_lg_emb)
                prev_lg_emb = self.lg_embedding(self.E_lg.index_select(index=lg_labels[:, idx], dim=0))
                scores = torch.cat([scores, scores_lg.unsqueeze(1)], dim=1)

        return scores

    def forward(self, prod_inputs, frag_inputs):
        prod_tensors, prod_scopes = prod_inputs
        frag_tensors, frag_scopes = frag_inputs

        prod_tensors = self.to_device(prod_tensors)
        frag_tensors = self.to_device(frag_tensors)

        prod_vecs, _ = self.encoder(prod_tensors, prod_scopes)
        frag_vecs, c_atom = self.encoder(frag_tensors, frag_scopes)
        frag_vecs_pad = torch.nn.utils.rnn.pad_sequence(frag_vecs, batch_first=True)

        return prod_vecs, frag_vecs_pad

    def get_saveables(self):
        saveables = {}
        saveables['config'] = self.config
        saveables['lg_vocab'] = self.lg_vocab
        saveables['encoder_name'] = self.encoder_name
        saveables['toggles'] = None if self.toggles == {} else self.toggles
        return saveables

    def to_device(self, tensors):
        """Converts all inputs to the device used."""
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device) for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def _compute_lg_stats(self, lg_logits, lg_labels, lengths):
        loss = self.lg_loss(lg_logits.view(-1, len(self.lg_vocab)), lg_labels.reshape(-1))
        acc_lg = get_accuracy_lg(lg_logits, lg_labels, lengths, device=self.device)
        return loss, acc_lg

    def train_step(self, prod_inputs, frag_inputs, lg_labels, lengths, **kwargs):
        prod_vecs, frag_vecs_pad = self(prod_inputs, frag_inputs)
        lg_labels = self.to_device(lg_labels)
        lg_logits = self._compute_lg_logits(frag_vecs_pad, prod_vecs=prod_vecs, lg_labels=lg_labels)

        lg_loss, lg_acc = self._compute_lg_stats(lg_logits, lg_labels, lengths)
        metrics = {'loss': lg_loss.item(), "accuracy": lg_acc.item()}
        return lg_loss, metrics

    def eval_step(self, prod_smi_batch: List[str],
                  core_edits_batch: List[List],
                  lg_label_batch: List[List],
                  rxn_classes: List[int] = None,
                  **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Eval step of the model.

        Parameters
        ----------
        prod_smi_batch: List[str],
            List of product smiles
        core_edits_batch: List[List],
            List of edits for each element in batch.
        lg_label_batch: List[List],
            Leaving groups for each element in the batch
        """
        acc_lg = 0.0

        for idx, prod_smi in enumerate(prod_smi_batch):
            if rxn_classes is None:
                labels = self.predict(prod_smi, core_edits_batch[idx])
            else:
                labels = self.predict(prod_smi, core_edits_batch[idx], rxn_class=rxn_classes[idx])
            if labels == lg_label_batch[idx]:
                acc_lg += 1.0

        metrics = {'loss': None, 'accuracy': acc_lg}
        return None, metrics

    def predict(self, prod_smi: str, core_edits: List, rxn_class: int = None):
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        core_edits: List,
            Edits associated with product molecule
        """
        if self.encoder_name == 'WLNEncoder':
            directed = False
        elif self.encoder_name == 'GraphFeatEncoder':
            directed = True

        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        with torch.no_grad():
            mol = Chem.MolFromSmiles(prod_smi)
            prod_graph = RxnElement(mol=Chem.Mol(mol), rxn_class=rxn_class)

            prod_inputs = pack_graph_feats([prod_graph], directed=directed,
                                           return_graphs=False, use_rxn_class=use_rxn_class)
            fragments = apply_edits_to_mol(Chem.Mol(mol), core_edits)
            tmp_frags = MultiElement(Chem.Mol(fragments)).mols

            if fragments is None:
                return []

            else:
                fragments = Chem.Mol()
                for mol in tmp_frags:
                    fragments = Chem.CombineMols(fragments, mol)

                frag_graph = MultiElement(mol=Chem.Mol(fragments), rxn_class=rxn_class)
                frag_inputs = pack_graph_feats([frag_graph], directed=directed,
                                                return_graphs=False, use_rxn_class=use_rxn_class)

                prod_vecs, frag_vecs_pad = self(prod_inputs, frag_inputs)

                lg_logits = self._compute_lg_logits(frag_vecs_pad, prod_vecs, lg_labels=None)

                _, preds = torch.max(lg_logits, dim=-1)
                preds = preds.squeeze(0)
                pred_labels = [self.lg_vocab.get_elem(pred.item()) for pred in preds]

                return pred_labels
