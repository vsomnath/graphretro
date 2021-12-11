import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import math
from typing import List, Dict, Tuple, Union

from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.layers import (AtomAttention, GraphFeatEncoder, WLNEncoder,
    GTransEncoder, LogitEncoder)
from seq_graph_retro.utils.torch import index_select_ND, build_mlp
from seq_graph_retro.utils.metrics import (get_accuracy_lg, get_accuracy_overall,
            get_accuracy_edits)
from seq_graph_retro.molgraph.mol_features import BOND_DELTAS, BOND_FLOATS

from seq_graph_retro.utils.parse import apply_edits_to_mol
from seq_graph_retro.utils.chem import get_mol
from seq_graph_retro.data.collate_fns import pack_graph_feats
from seq_graph_retro.molgraph.rxn_graphs import MultiElement, RxnElement


class SingleEditShared(nn.Module):

    def __init__(self,
                 config: Dict,
                 lg_vocab: Vocab,
                 tensor_file: str,
                 encoder_name: str,
                 device: str = 'cpu',
                 toggles: Dict = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        config: Dict,
            Config for all sub-modules and self
        lg_vocab: Vocab
            Vocabulary of leaving groups
        encoder_name: str,
            Name of the encoder network
        toggles: Dict, default None
            Optional toggles for the model. Useful for ablation studies
        device: str,
            Device to run the model on.
        """
        super(SingleEditShared, self).__init__(**kwargs)
        self.config = config
        self.lg_vocab = lg_vocab
        self.tensor_file = tensor_file
        self.lg_tensors, self.lg_scopes = torch.load(self.tensor_file)
        self.toggles = toggles if toggles is not None else {}
        self.encoder_name = encoder_name
        self.device = device

        self.E_lg = torch.eye(len(self.lg_vocab)).to(device)

        self._build_layers()
        self._build_losses()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
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

        elif self.encoder_name == 'GTransEncoder':
            self.encoder = GTransEncoder(node_fdim=config['n_atom_feat'],
                                         edge_fdim=config['n_bond_feat'],
                                         rnn_type=config['rnn_type'],
                                         hsize=config['mpn_size'], depth=config['depth'],
                                         n_heads=config['n_heads'], bias=config['bias'],
                                         n_mt_blocks=config['n_mt_blocks'],
                                         dropout_p=config['dropout_mpn'],
                                         res_connection=self.toggles.get("use_res", False))

        if self.toggles.get('use_attn', False):
            self.attn_layer = AtomAttention(n_bin_feat=config['n_bin_feat'],
                                            hsize=config['mpn_size'],
                                            n_heads=config['n_heads'],
                                            bias=config['bias'])

        bond_score_in_dim = 2 * config['mpn_size']
        unimol_score_in_dim = config['mpn_size']

        if self.toggles.get("use_prod", False):
            add_dim = config['mpn_size']
            if self.toggles.get("use_concat", False):
                add_dim *= config['depth']
            bond_score_in_dim += add_dim
            unimol_score_in_dim += add_dim

        self.bond_score = build_mlp(in_dim=bond_score_in_dim,
                               h_dim=config['mlp_size'],
                               out_dim=config['bs_outdim'],
                               dropout_p=config['dropout_mlp'])
        self.unimol_score = build_mlp(in_dim=unimol_score_in_dim,
                                      out_dim=1, h_dim=config['mlp_size'],
                                      dropout_p=config['dropout_mlp'])

        if self.toggles.get("propagate_logits", False):
            hsize = config['mpn_size'] // 4
            self.bond_label_mpn = LogitEncoder(rnn_type=config['rnn_type'],
                                               edge_fdim=config['bond_label_feat'],
                                               node_fdim=config['n_bond_feat'],
                                               hsize=hsize, depth=config['depth'] // 3,
                                               dropout_p=config['dropout_mpn'],
                                               outdim=config['bs_outdim'])

        ### Build components for leaving group classifier
        self.base_embeddings = nn.Parameter(torch.FloatTensor(4, config['embed_size']))
        nn.init.kaiming_uniform_(self.base_embeddings, a=math.sqrt(5))
        self.W_proj = nn.Linear(config['mpn_size'], config['embed_size'],
                                bias=config['embed_bias'])

        lg_score_in_dim = 2 * config['mpn_size']
        if self.toggles.get('use_prev_pred', False):
            lg_score_in_dim += config['embed_size']

        self.lg_score = build_mlp(in_dim=lg_score_in_dim,
                                  h_dim=config['mlp_size'],
                                  out_dim=len(self.lg_vocab),
                                  dropout_p=config['dropout_mlp'])

    def _build_losses(self) -> None:
        """Builds losses associated with the model."""
        if self.config['edit_loss'] == 'sigmoid':
            self.edit_loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.edit_loss = nn.CrossEntropyLoss(reduction='none')
        self.lg_loss = nn.CrossEntropyLoss(ignore_index=self.lg_vocab["<pad>"])

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True) if tensor is not None else None
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def get_saveables(self):
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['lg_vocab'] = self.lg_vocab
        saveables['tensor_file'] = self.tensor_file
        saveables['encoder_name'] = self.encoder_name
        saveables['toggles'] = None if self.toggles == {} else self.toggles
        return saveables

    def _compute_edit_logits(self, graph_tensors: Tuple[torch.Tensor], scopes: Tuple[List],
                             bg_inputs: Tuple[torch.Tensor, List] = None,
                             ha: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """Computes edits logits for the model.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tensors representing a batch of graphs. Includes atom and bond features,
            and bond and atom neighbors
        scopes: Tuple[List],
            Scopes is composed of atom and bond scopes, which keep track of atom
            and bond indices for each molecule in the 2D feature list
        """
        atom_scope, bond_scope = scopes

        c_mol, c_atom = self.encoder(graph_tensors, scopes)
        if self.toggles.get('use_attn', False):
            c_mol, c_atom_att = self.attn_layer(c_atom, scopes)
            c_atom_starts = index_select_ND(c_atom_att, dim=0, index=graph_tensors[-1][:, 0])
            c_atom_ends = index_select_ND(c_atom_att, dim=0, index=graph_tensors[-1][:, 1])

        else:
            c_atom_starts = index_select_ND(c_atom, dim=0, index=graph_tensors[-1][:, 0])
            c_atom_ends = index_select_ND(c_atom, dim=0, index=graph_tensors[-1][:, 1])

        sum_bonds = c_atom_starts + c_atom_ends
        diff_bonds = torch.abs(c_atom_starts - c_atom_ends)
        bond_score_inputs = torch.cat([sum_bonds, diff_bonds], dim=1)
        atom_score_inputs = c_atom.clone()

        if self.toggles.get("use_prod", False):
            atom_scope, bond_scope = scopes
            mol_exp_atoms = torch.cat([c_mol[idx].expand(le, -1)
                                    for idx, (st, le) in enumerate(atom_scope)], dim=0)
            mol_exp_bonds = torch.cat([c_mol[idx].expand(le, -1)
                                    for idx, (st, le) in enumerate(bond_scope)], dim=0)
            mol_exp_atoms = torch.cat([c_mol.new_zeros(1, c_mol.shape[-1]), mol_exp_atoms], dim=0)
            mol_exp_bonds = torch.cat([c_mol.new_zeros(1, c_mol.shape[-1]), mol_exp_bonds], dim=0)
            assert len(mol_exp_atoms) == len(atom_score_inputs)
            assert len(mol_exp_bonds) == len(bond_score_inputs)

            bond_score_inputs = torch.cat([bond_score_inputs, mol_exp_bonds], dim=-1)
            atom_score_inputs = torch.cat([atom_score_inputs, mol_exp_atoms], dim=-1)

        bond_logits = self.bond_score(bond_score_inputs)
        unimol_logits = self.unimol_score(atom_score_inputs)

        if self.toggles.get("propagate_logits", False):
            bg_tensors, bg_scope = bg_inputs
            assert len(bond_logits) == len(bg_tensors[0])
            bond_logits = self.bond_label_mpn(bond_logits, bg_tensors, mask=None)
            edit_logits = [torch.cat([bond_logits[st_b: st_b+le_b].flatten(),
                                     unimol_logits[st_a: st_a+le_a].flatten()], dim=-1)
                           for ((st_a, le_a), (st_b, le_b)) in zip(*(atom_scope, bond_scope))]
            return c_mol, edit_logits, None

        edit_logits = [torch.cat([bond_logits[st_b: st_b+le_b].flatten(),
                                 unimol_logits[st_a: st_a+le_a].flatten()], dim=-1)
                       for ((st_a, le_a), (st_b, le_b)) in zip(*(atom_scope, bond_scope))]

        return c_mol, edit_logits, None

    def _compute_lg_step(self, graph_vecs: torch.Tensor, prod_vecs: torch.Tensor,
                         prev_embed: torch.Tensor = None) -> torch.Tensor:
        """Run a single step of leaving group addition.

        Parameters
        ----------
        graph_vecs: torch.Tensor,
            Graph vectors for fragments at that step
        prod_vecs: torch.Tensor,
            Graph vectors for products
        prev_embed: torch.Tensor, default None,
            Embedding of previous leaving group.
        """
        if not self.training:
            lg_tensors = tuple([tensor.clone() for tensor in self.lg_tensors])
            lg_tensors = self.to_device(lg_tensors)
            cmol, _ = self.encoder(lg_tensors, self.lg_scopes)

            self.lg_embedding = torch.cat([self.base_embeddings, self.W_proj(cmol)], dim=0)

        if self.toggles.get('use_prev_pred', False):
            if prev_embed is None:
                init_state = torch.zeros(graph_vecs.size(0), len(self.lg_vocab), device=self.device)
                init_state[:, 0] = 1
                prev_lg_emb = self.lg_embedding.index_select(index=torch.argmax(init_state, dim=-1), dim=0)
            else:
                prev_lg_emb = prev_embed

        if self.toggles.get('use_prev_pred', False):
            scores_lg = self.lg_score(torch.cat([prev_lg_emb, prod_vecs, graph_vecs], dim=-1))
        else:
            scores_lg = self.lg_score(torch.cat([prod_vecs, graph_vecs], dim=-1))

        lg_embed = self.lg_embedding.index_select(index=torch.argmax(scores_lg, dim=-1), dim=0)
        return scores_lg, lg_embed

    def _compute_lg_logits(self, graph_vecs_pad, prod_vecs, lg_labels=None) -> torch.Tensor:
        """Computes leaving group logits.

        Parameters
        ----------
        graph_vecs_pad: torch.Tensor,
            Graph vectors for fragments
        prod_vecs: torch.Tensor,
            Graph vectors for products
        lg_labels: torch.Tensor, default None,
            Correct leaving group indices. Used in teacher forcing if not None.
            Else maximum from previous case is used.
        """
        scores = torch.tensor([], device=self.device)
        prev_lg_emb = None

        if lg_labels is None:
            for idx in range(graph_vecs_pad.size(1)):
                scores_lg, prev_lg_emb = self._compute_lg_step(graph_vecs_pad[:, idx], prod_vecs, prev_embed=prev_lg_emb)
                scores = torch.cat([scores, scores_lg.unsqueeze(1)], dim=1)

        else:
            for idx in range(graph_vecs_pad.size(1)):
                scores_lg, _ = self._compute_lg_step(graph_vecs_pad[:, idx], prod_vecs, prev_embed=prev_lg_emb)
                prev_lg_emb = self.lg_embedding.index_select(index=lg_labels[:, idx], dim=0)
                scores = torch.cat([scores, scores_lg.unsqueeze(1)], dim=1)

        return scores

    def _compute_edit_stats(self, edit_logits: List[torch.Tensor],
                            edit_labels: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Computes edit loss and accuracy given the labels.

        Parameters
        ----------
        edit_logits: List[torch.Tensor]
            Edit logits for all examples in batch
        edit_labels: List[torch.Tensor]
            Edit labels for all examples in batch
        """
        if self.config['edit_loss'] == 'sigmoid':
            loss_batch = [self.edit_loss(edit_logits[i].unsqueeze(0), edit_labels[i].unsqueeze(0)).sum()
                          for i in range(len(edit_logits))]

        else:
            loss_batch = [self.edit_loss(edit_logits[i].unsqueeze(0),
                                            torch.argmax(edit_labels[i]).unsqueeze(0).long()).sum()
                          for i in range(len(edit_logits))]

        loss = torch.stack(loss_batch, dim=0).mean()
        accuracy = get_accuracy_edits(edit_logits, edit_labels)
        return loss, accuracy

    def _compute_lg_stats(self, lg_logits: torch.Tensor,
                          lg_labels: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.tensor]:
        """
        Computes leaving group addition loss and accuracy given logits and labels

        Parameters
        ----------
        lg_logits: torch.Tensor,
            Leaving group logits tensor
        lg_labels: torch.Tensor,
            Leaving group labels tensor
        lengths: torch.Tensor,
            True number of fragments in every example
        """
        loss = self.lg_loss(lg_logits.view(-1, len(self.lg_vocab)), lg_labels.reshape(-1))
        acc_lg = get_accuracy_lg(lg_logits, lg_labels, lengths, device=self.device)
        return loss, acc_lg

    def forward(self, prod_inputs: Tuple[torch.Tensor, List[int]],
                bg_inputs: Tuple[torch.Tensor, List],
                frag_inputs: Tuple[torch.Tensor, List[List]]) -> Tuple[torch.Tensor]:
        """
        Forward propagation step.

        Parameters
        ----------
        prod_inputs: Tuple[torch.Tensor, List[int]],
            Consists of product tensors and scopes
        frag_inputs: Tuple[torch.Tensor, List[List]],
            Consists of fragment tensors and scopes
        """
        prod_tensors, prod_scopes = prod_inputs
        frag_tensors, frag_scopes = frag_inputs

        prod_tensors = self.to_device(prod_tensors)
        frag_tensors = self.to_device(frag_tensors)

        if bg_inputs is not None:
            bg_tensors, bg_scope = bg_inputs
            bg_tensors = self.to_device(bg_tensors)
            bg_inputs = (bg_tensors, bg_scope)

        prod_vecs, edit_logits, _ = self._compute_edit_logits(prod_tensors, prod_scopes, bg_inputs, ha=None)
        frag_vecs, c_atom = self.encoder(frag_tensors, frag_scopes)
        frag_vecs_pad = torch.nn.utils.rnn.pad_sequence(frag_vecs, batch_first=True)

        return prod_vecs, edit_logits, frag_vecs_pad

    def train_step(self, prod_inputs: Tuple[torch.Tensor, List],
                   bg_inputs: Tuple[torch.Tensor, List],
                   frag_inputs: Tuple[torch.Tensor, List], edit_labels: List[torch.Tensor],
                   lg_labels: torch.Tensor, lengths: torch.Tensor, **kwargs):
        """
        Train step of the model

        Parameters
        ----------
        prod_inputs: Tuple[torch.Tensor, List]
            List of prod_tensors for edit sequence
        frag_inputs: Tuple[torch.Tensor, List[List]],
            Consists of fragment tensors and scopes
        edit_labels: List[torch.Tensor],
            List of edit labels for each step of the sequence. The last label is
            a done label
        lg_labels: torch.Tensor,
            Leaving group labels tensor
        seq_mask: torch.Tensor,
            Seq mask capturing sequence lengths of different batch elements
        lengths: torch.Tensor,
            True number of fragments in every example
        """
        lg_tensors = self.to_device(self.lg_tensors)
        cmol, _ = self.encoder(lg_tensors, self.lg_scopes)

        self.lg_embedding = torch.cat([self.base_embeddings, self.W_proj(cmol)], dim=0)

        prod_vecs, edit_logits, frag_vecs_pad = self(prod_inputs=prod_inputs,
                                                     bg_inputs=bg_inputs,
                                                     frag_inputs=frag_inputs)
        lg_labels = self.to_device(lg_labels)
        edit_labels = self.to_device(edit_labels)
        lg_logits = self._compute_lg_logits(frag_vecs_pad, prod_vecs=prod_vecs, lg_labels=lg_labels)

        edit_loss, edit_acc = self._compute_edit_stats(edit_logits, edit_labels)
        lg_loss, lg_acc = self._compute_lg_stats(lg_logits, lg_labels, lengths)
        accuracy = get_accuracy_overall(edit_logits, lg_logits, edit_labels, lg_labels, lengths, device=self.device)

        loss = self.config['lam_edits'] * edit_loss +  \
               self.config['lam_lg'] * lg_loss

        metrics = {'loss': loss.item(),
                   'edit_loss': edit_loss.item(),
                   'lg_loss': lg_loss.item(),
                   'accuracy': accuracy.item(),
                   'edit_acc': edit_acc.item(),
                   'lg_acc': lg_acc.item()}

        return loss, metrics

    def eval_step(self, prod_smi_batch: List[str],
                  core_edits_batch: List[List],
                  lg_label_batch: List[List],
                  rxn_classes: List[int] = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
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

        acc_overall = 0.0
        acc_lg = 0.0
        acc_edits = 0.0

        for idx, prod_smi in enumerate(prod_smi_batch):
            if rxn_classes is None:
                edits, labels = self.predict(prod_smi)
            else:
                edits, labels = self.predict(prod_smi, rxn_class=rxn_classes[idx])
            if set([edits]) == set(core_edits_batch[idx]) and labels == lg_label_batch[idx]:
                acc_overall += 1.0

            if set([edits]) == set(core_edits_batch[idx]):
                acc_edits += 1.0

            if labels == lg_label_batch[idx]:
                acc_lg += 1.0

        metrics = {'loss': None, 'edit_acc': acc_edits, 'lg_acc': acc_lg, 'accuracy': acc_overall}
        return None, metrics

    def predict(self, prod_smi: str, rxn_class: int = None) -> Tuple[List]:
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
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
            prod_tensors, prod_scopes = pack_graph_feats([prod_graph],
                                                          directed=directed, return_graphs=False,
                                                          use_rxn_class=use_rxn_class)
            bg_inputs = None
            if self.toggles.get("propagate_logits", False):
                bg_inputs = tensorize_bond_graphs([prod_graph], directed=directed,
                                                   use_rxn_class=use_rxn_class)
                bg_tensors, bg_scope = bg_inputs
                bg_tensors = self.to_device(bg_tensors)
                bg_inputs = (bg_tensors, bg_scope)

            prod_tensors = self.to_device(prod_tensors)
            prod_vecs, edit_logits, _ = self._compute_edit_logits(prod_tensors, prod_scopes, bg_inputs)
            idx = torch.argmax(edit_logits[0])
            val = edit_logits[0][idx]

            if self.config['bs_outdim'] > 1:
                max_bond_idx = mol.GetNumBonds() * len(BOND_FLOATS)
            elif self.config['bs_outdim'] == 1:
                max_bond_idx = mol.GetNumBonds()

            if self.toggles.get('use_h_labels', False):

                if idx.item() < max_bond_idx:

                    if self.config['bs_outdim'] > 1:
                        bond_logits = edit_logits[0][:mol.GetNumBonds() * len(BOND_FLOATS)]
                        bond_logits = bond_logits.reshape(mol.GetNumBonds(), len(BOND_FLOATS))
                        idx_tensor = torch.where(bond_logits == val)

                        idx_tensor = [indices[-1] for indices in idx_tensor]

                        bond_idx, bo_idx = idx_tensor[0].item(), idx_tensor[1].item()
                        new_bo = BOND_FLOATS[bo_idx]
                        #bond_delta = list(BOND_DELTAS.keys())[list(BOND_DELTAS.values()).index(delta_idx)]
                        a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                        a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                        a1, a2 = sorted([a1, a2])
                        bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()
                        #new_bo = bo + bond_delta

                        edit = f"{a1}:{a2}:{bo}:{new_bo}"

                    elif self.config['bs_outdim'] == 1:
                        bond_idx = idx.item()
                        a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                        a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                        a1, a2 = sorted([a1, a2])
                        bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()

                        edit = f"{a1}:{a2}:{bo}:{0.0}"

                    else:
                        pass

                else:
                    h_logits = edit_logits[0][max_bond_idx:]
                    assert len(h_logits) == mol.GetNumAtoms()

                    atom_idx = idx.item() - max_bond_idx
                    a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()

                    edit = f"{a1}:{0}:{1.0}:{0.0}"

            else:

                if idx.item() == len(edit_logits) - 1:
                    pass

                elif self.config['bs_outdim'] > 1:
                    bond_logits = edit_logits[0][:mol.GetNumBonds() * len(BOND_DELTAS)].reshape(mol.GetNumBonds(), len(BOND_DELTAS))
                    idx_tensor = torch.where(bond_logits == val)

                    idx_tensor = [indices[-1] for indices in idx_tensor]

                    bond_idx, delta_idx = idx_tensor[0].item(), idx_tensor[1].item()
                    bond_delta = list(BOND_DELTAS.keys())[list(BOND_DELTAS.values()).index(delta_idx)]
                    a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                    a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                    a1, a2 = sorted([a1, a2])
                    bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()
                    new_bo = bo + bond_delta

                    edit = f"{a1}:{a2}:{bo}:{new_bo}"

                elif self.config['bs_outdim'] == 1:
                    bond_idx = idx.item()
                    a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                    a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                    a1, a2 = sorted([a1, a2])
                    bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()

                    edit = f"{a1}:{a2}:{bo}:{0.0}"

                else:
                    pass

            try:
                fragments = apply_edits_to_mol(get_mol(prod_smi), [edit])
                tmp_frags = MultiElement(mol=fragments).mols
                fragments = Chem.Mol()
                for mol in tmp_frags:
                    fragments = Chem.CombineMols(fragments, mol)
                frag_graph = MultiElement(mol=fragments, rxn_class=rxn_class)

                frag_tensors, frag_scopes = pack_graph_feats(graph_batch=[frag_graph], directed=directed,
                                                     return_graphs=False, use_rxn_class=use_rxn_class)

            except:
                return edit, []

            frag_tensors = self.to_device(frag_tensors)
            frag_vecs, _ = self.encoder(frag_tensors, frag_scopes)

            frag_vecs_pad = torch.nn.utils.rnn.pad_sequence(frag_vecs, batch_first=True)
            lg_logits = self._compute_lg_logits(frag_vecs_pad, prod_vecs, lg_labels=None)

            _, preds = torch.max(lg_logits, dim=-1)
            preds = preds.squeeze(0)
            pred_labels = [self.lg_vocab.get_elem(pred.item()) for pred in preds]

        return edit, pred_labels
