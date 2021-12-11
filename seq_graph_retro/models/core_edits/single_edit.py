import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from typing import List, Dict, Tuple, Union

from seq_graph_retro.layers import (AtomAttention, GraphFeatEncoder, WLNEncoder,
                            LogitEncoder, GTransEncoder)
from seq_graph_retro.utils.torch import index_select_ND, build_mlp
from seq_graph_retro.utils.metrics import get_accuracy_edits
from seq_graph_retro.molgraph.mol_features import BOND_FLOATS

from seq_graph_retro.data.collate_fns import pack_graph_feats, tensorize_bond_graphs
from seq_graph_retro.molgraph.rxn_graphs import RxnElement


class SingleEdit(nn.Module):
    """Model to predict the edit labels associated with a product molecule.
    Supports only single edits."""

    def __init__(self,
                 config: Dict,
                 encoder_name: str,
                 toggles: Dict = None,
                 device: str = 'cpu',
                 **kwargs) -> None:
        """
        Parameters
        ----------
        config: Dict,
            Configuration for layers in model.
        encoder_name: str,
            Name of the encoder used. Allows message passing in directed or
            undirected format
        toggles: Dict, default None
            Optional toggles for the model. Useful for ablation studies
        device: str,
            Device to run the model on.
        """
        super(SingleEdit, self).__init__(**kwargs)
        self.config = config
        self.encoder_name = encoder_name
        self.toggles = toggles if toggles is not None else {}
        self.device = device

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


    def _build_losses(self) -> None:
        """Builds losses associated with the model."""
        if self.config['edit_loss'] == 'sigmoid':
            config = self.config
            self.edit_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.FloatTensor([config['pos_weight']]))
        elif self.config['edit_loss'] == 'softmax':
            self.edit_loss = nn.CrossEntropyLoss(reduction='none')

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

    def _compute_edit_logits(self, graph_tensors: Tuple[torch.Tensor],
                             scopes: Tuple[List],  bg_inputs: torch.Tensor = None,
                             ha: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        Computes the edit logits.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tensors representing a batch of graphs. Includes atom and bond features,
            and bond and atom neighbors
        scopes: Tuple[List],
            Scopes is composed of atom and bond scopes, which keep track of atom
            and bond indices for each molecule in the 2D feature list
        ha: torch.Tensor, default None
            Hidden states of atoms in the molecule
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

    def forward(self, graph_tensors: Tuple[torch.Tensor], scopes: Tuple[List],
                bg_inputs = None) -> Tuple[torch.Tensor]:
        """Forward pass

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tensors representing a batch of graphs. Includes atom and bond features,
            and bond and atom neighbors
        scopes: Tuple[List],
            Scopes is composed of atom and bond scopes, which keep track of atom
            and bond indices for each molecule in the 2D feature list
        """
        graph_tensors = self.to_device(graph_tensors)
        if bg_inputs is not None:
            bg_tensors, bg_scope = bg_inputs
            bg_tensors = self.to_device(bg_tensors)
            bg_inputs = (bg_tensors, bg_scope)
        c_mol, edit_logits, _ = self._compute_edit_logits(graph_tensors, scopes,
                                                       ha=None, bg_inputs=bg_inputs)
        return c_mol, edit_logits

    def train_step(self, graph_tensors: Tuple[torch.Tensor],
                   scopes: Tuple[List], bg_inputs: Tuple[Tuple[torch.Tensor], Tuple[List]],
                   edit_labels: List[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Train step of the model.

        Parameters
        ----------
        graph_tensors: Tuple[torch.Tensor],
            Tensors representing a batch of graphs. Includes atom and bond features,
            and bond and atom neighbors
        scopes: Tuple[List],
            Scopes is composed of atom and bond scopes, which keep track of atom
            and bond indices for each molecule in the 2D feature list
        edit_labels: List[torch.Tensor]
            Edit labels for given batch of molecules
        """
        edit_labels = self.to_device(edit_labels)

        prod_vecs, edit_logits = self(graph_tensors, scopes, bg_inputs)
        if self.config['edit_loss'] == 'sigmoid':
            loss_batch = [self.edit_loss(edit_logits[i].unsqueeze(0), edit_labels[i].unsqueeze(0)).sum()
                          for i in range(len(edit_logits))]

        elif self.config['edit_loss'] == 'softmax':
            loss_batch = [self.edit_loss(edit_logits[i].unsqueeze(0),
                                            torch.argmax(edit_labels[i]).unsqueeze(0).long()).sum()
                          for i in range(len(edit_logits))]
        else:
            raise ValueError()

        loss = torch.stack(loss_batch, dim=0).mean()
        accuracy = get_accuracy_edits(edit_logits, edit_labels)
        metrics = {'loss': loss.item(), 'accuracy': accuracy.item()}
        return loss, metrics

    def eval_step(self, prod_smi_batch: List[str],
                  core_edits_batch: List[str],
                  rxn_classes: List[int] = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Eval step of the model.

        Parameters
        ----------
        prod_smi_batch: List[str],
            List of product smiles
        core_edits_batch: List[List]:
            List of edits for each element in batch
        rxn_classes: List[int], default None,
            List of reaction classes
        """
        loss = None
        accuracy = 0.0

        for idx, prod_smi in enumerate(prod_smi_batch):
            if rxn_classes is None:
                edits = self.predict(prod_smi)
            else:
                edits = self.predict(prod_smi, rxn_class=rxn_classes[idx])
            if set(edits) == set(core_edits_batch[idx]):
                accuracy += 1.0

        metrics = {'loss': None, 'accuracy': accuracy}
        return loss, metrics

    def predict(self, prod_smi: str, rxn_class: int = None) -> List:
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None,
            Reaction class
        """
        if self.encoder_name == 'WLNEncoder':
            directed = False
        elif self.encoder_name == 'GraphFeatEncoder':
            directed = True
        elif self.encoder_name == 'GTransEncoder':
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
            _, edit_logits = self(prod_tensors, prod_scopes, bg_inputs)
            idx = torch.argmax(edit_logits[0])
            val = edit_logits[0][idx]

            if self.config['bs_outdim'] > 1:
                max_bond_idx = mol.GetNumBonds() * len(BOND_FLOATS)
            elif self.config['bs_outdim'] == 1:
                max_bond_idx = mol.GetNumBonds()

            if idx.item() < max_bond_idx:
                if self.config['bs_outdim'] > 1:
                    bond_logits = edit_logits[0][:mol.GetNumBonds() * len(BOND_FLOATS)]
                    bond_logits = bond_logits.reshape(mol.GetNumBonds(), len(BOND_FLOATS))
                    idx_tensor = torch.where(bond_logits == val)

                    idx_tensor = [indices[-1] for indices in idx_tensor]

                    bond_idx, bo_idx = idx_tensor[0].item(), idx_tensor[1].item()
                    a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                    a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                    a1, a2 = sorted([a1, a2])
                    bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()
                    new_bo = BOND_FLOATS[bo_idx]

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

        return [edit]

    def get_kthedit(self, mol, edit_logits, k=1):
        values, indices = edit_logits[0].topk(k=k, dim=0)
        idx = indices[k-1]
        val = values[k-1]

        if self.config['bs_outdim'] > 1:
            max_bond_idx = mol.GetNumBonds() * len(BOND_FLOATS)
        elif self.config['bs_outdim'] == 1:
            max_bond_idx = mol.GetNumBonds()

        if idx.item() < max_bond_idx:
            if self.config['bs_outdim'] > 1:
                bond_logits = edit_logits[0][:mol.GetNumBonds() * len(BOND_FLOATS)]
                bond_logits = bond_logits.reshape(mol.GetNumBonds(), len(BOND_FLOATS))
                idx_tensor = torch.where(bond_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]

                bond_idx, bo_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                a1, a2 = sorted([a1, a2])
                bo = mol.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()
                new_bo = BOND_FLOATS[bo_idx]

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

        return [edit]

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['encoder_name'] = self.encoder_name
        saveables['toggles'] = None if self.toggles == {} else self.toggles
        return saveables
