import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from typing import List, Dict, Tuple, Union

from seq_graph_retro.layers import AtomAttention, GraphFeatEncoder, WLNEncoder
from seq_graph_retro.utils.torch import index_select_ND, build_mlp
from seq_graph_retro.utils.metrics import get_edit_seq_accuracy
from seq_graph_retro.molgraph.mol_features import BOND_FLOATS

from seq_graph_retro.utils.parse import apply_edits_to_mol
from seq_graph_retro.data.collate_fns import pack_graph_feats
from seq_graph_retro.molgraph.rxn_graphs import RxnElement


class MultiEdit(nn.Module):
    """Model to predict the edit labels associated with a product molecule.
    Supports multiple edits."""

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
        super(MultiEdit, self).__init__(**kwargs)
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
            self.encoder = GraphFeatEncoder(n_atom_feat=config['n_atom_feat'],
                                            n_bond_feat=config['n_bond_feat'],
                                            rnn_type=config['rnn_type'],
                                            hsize=config['mpn_size'],
                                            depth=config['depth'],
                                            dropout_p=config['dropout_mpn'])

        elif self.encoder_name == 'WLNEncoder':
            self.encoder = WLNEncoder(n_atom_feat=config['n_atom_feat'],
                                      n_bond_feat=config['n_bond_feat'],
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

        bond_score_in_dim = 2 * config['mpn_size']
        unimol_score_in_dim = config['mpn_size']

        self.W_vv = nn.Linear(config['mpn_size'], config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'], config['mpn_size'], bias=False)

        self.atom_proj = nn.Linear(in_features=config['mpn_size'], out_features= 2*config['mpn_size'])

        self.bond_score = build_mlp(in_dim=bond_score_in_dim,
                                    h_dim=config['mlp_size'],
                                    out_dim=config['bs_outdim'],
                                    dropout_p=config['dropout_mlp'])
        self.bond_score_in_dim = bond_score_in_dim

        if self.toggles.get('use_h_labels', False):
            self.unimol_score = build_mlp(in_dim=unimol_score_in_dim,
                                          out_dim=1, h_dim=config['mlp_size'],
                                          dropout_p=config['dropout_mlp'])
        else:
            self.unimol_score = build_mlp(in_dim=config['mpn_size'],
                                          out_dim=1, h_dim=config['mlp_size'],
                                          dropout_p=config['dropout_mlp'])

        self.done_score = build_mlp(in_dim=config['mpn_size'],
                                    out_dim=1,
                                    h_dim=config['mlp_size'],
                                    dropout_p=config['dropout_mlp'])

    def _build_losses(self) -> None:
        """Builds losses associated with the model."""
        if self.config['edit_loss'] == 'sigmoid':
            self.edit_loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
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
            tensors = [tensor.to(self.device, non_blocking=True) for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def _compute_edit_logits(self, prod_tensors: Tuple[torch.Tensor],
                             prod_scopes: Tuple[List], ha: torch.Tensor = None,
                             **kwargs) -> Tuple[torch.Tensor]:
        """Computes the edit logits given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        ha: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)
        atom_scope, bond_scope = prod_scopes
        if ha is None:
            bs = len(atom_scope)
            n_atoms = prod_tensors[0].size(0)
            ha = torch.zeros(n_atoms, self.config['hsize'], device=self.device)

        c_mol, c_atom = self.encoder(prod_tensors, prod_scopes)
        assert c_atom.shape == ha.shape

        if self.toggles.get('use_attn', False):
            c_mol, c_atom = self.attn_layer(c_atom, prod_scopes)

        ha = F.relu(self.W_vv(ha) + self.W_vc(c_atom))
        ha[0] = 0
        hm = torch.stack([ha[st: st + le].sum(dim=0) for st, le in atom_scope])

        c_atom_starts = index_select_ND(ha, dim=0, index=prod_tensors[-1][:, 0])
        c_atom_ends = index_select_ND(ha, dim=0, index=prod_tensors[-1][:, 1])

        bond_score_inputs = torch.cat([c_atom_starts, c_atom_ends], dim=-1)
        atom_score_inputs = ha.clone()
        bond_logits = self.bond_score(bond_score_inputs)

        if self.toggles.get('use_h_labels', False):
            unimol_logits = self.unimol_score(atom_score_inputs)
            done_logits = self.done_score(hm)
            edit_logits = [torch.cat([bond_logits[st_b: st_b+le_b].flatten(),
                                     unimol_logits[st_a: st_a+le_a].flatten(), done_logits[idx]], dim=-1)
                           for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]
        else:
            unimol_logits = self.unimol_score(atom_score_inputs)
            done_logits = self.done_score(hm)
            edit_logits = [torch.cat([bond_logits[idx].flatten(),
                                     unimol_logits[idx], done_logits[idx]], dim=-1)
                           for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]

        if self.toggles.get('use_attn', False):
            return c_mol, edit_logits, ha
        else:
            return c_mol, edit_logits, ha

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]],
                seq_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        seq_mask: torch.Tensor,
            Seq mask capturing sequence lengths of different batch elements.
        """
        max_seq_len = len(prod_seq_inputs)
        assert len(prod_seq_inputs[0]) == 2

        ha = None
        seq_edit_logits = []

        for idx in range(max_seq_len):
            prod_tensors, prod_scopes = prod_seq_inputs[idx]
            if idx == 0:
                c_mol, edit_logits, ha = self._compute_edit_logits(prod_tensors,
                                            prod_scopes, ha=ha)
            else:
                _, edit_logits, ha = self._compute_edit_logits(prod_tensors, prod_scopes, ha=ha)
            seq_edit_logits.append(edit_logits)

        return c_mol, seq_edit_logits

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

    def _compute_edit_stats(self, seq_edit_logits: List[List[torch.Tensor]],
                            seq_edit_labels: List[List[torch.Tensor]],
                            seq_mask: torch.Tensor) -> Tuple[torch.Tensor]:
        """Computes the edit loss and accuracy given the logits and labels.

        Parameters
        ----------
        seq_edit_logits: List[List[torch.Tensor]],
            List of logits for each step in the edit sequence
        seq_edit_labels: List[List[torch.Tensor]],
            List of edit labels for each step of the sequence. The last label is
            a done label
        seq_mask: torch.Tensor,
            Seq mask capturing sequence lengths of different batch elements.
        """
        max_seq_len, bs = seq_mask.size()

        seq_loss = []

        for idx in range(max_seq_len):
            edit_labels_idx = self.to_device(seq_edit_labels[idx])
            if self.config['edit_loss'] == 'sigmoid':
                loss_batch = [seq_mask[idx][i] * self.edit_loss(seq_edit_logits[idx][i].unsqueeze(0),
                                                                edit_labels_idx[i].unsqueeze(0)).sum()
                              for i in range(bs)]
            else:
                loss_batch = [seq_mask[idx][i] * self.edit_loss(seq_edit_logits[idx][i].unsqueeze(0),
                                                             torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()).sum()
                              for i in range(bs)]
            loss = torch.stack(loss_batch, dim=0).mean()
            seq_loss.append(loss)

        seq_loss = torch.stack(seq_loss).mean()
        accuracy = get_edit_seq_accuracy(seq_edit_logits, seq_edit_labels, seq_mask)
        return seq_loss, accuracy

    def train_step(self, seq_tensors: List[Tuple[torch.Tensor]],
                   seq_labels: List[torch.Tensor],
                   seq_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Train step of the model.

        Parameters
        ----------
        seq_tensors: List[Tuple[torch.Tensor]],
            List of tensors for each step in the edit sequence
        seq_labels: List[torch.Tensor],
            List of edit labels for each step of the sequence. The last label is
            a done label
        seq_mask: torch.Tensor,
            Seq mask capturing sequence lengths of different batch elements.
        """
        seq_mask = self.to_device(seq_mask)
        prod_vecs, seq_edit_logits = self(seq_tensors, seq_mask)
        seq_loss, seq_acc = self._compute_edit_stats(seq_edit_logits, seq_labels, seq_mask)
        metrics = {'loss': seq_loss.item(), 'accuracy': seq_acc.item()}
        return seq_loss, metrics

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 6) -> List:
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 6
            Max number of edit steps allowed
        """
        if self.encoder_name == 'WLNEncoder':
            directed = False
        elif self.encoder_name == 'GraphFeatEncoder':
            directed = True

        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        done = False
        steps = 0
        edits = []
        ha = None

        with torch.no_grad():
            products = Chem.MolFromSmiles(prod_smi)

            prod_graph = RxnElement(mol=Chem.Mol(products), rxn_class=rxn_class)
            prod_tensors, prod_scopes = pack_graph_feats([prod_graph],
                                                          directed=directed, return_graphs=False,
                                                          use_rxn_class=use_rxn_class)

            while not done and steps <= max_steps:
                _, edit_logits, ha = self._compute_edit_logits(prod_tensors,
                                                            prod_scopes, ha=ha)
                idx = torch.argmax(edit_logits[0])
                val = edit_logits[0][idx]

                if self.config['bs_outdim'] > 1:
                    max_bond_idx = products.GetNumBonds() * len(BOND_FLOATS)
                elif self.config['bs_outdim'] == 1:
                    max_bond_idx = products.GetNumBonds()

                if self.toggles.get('use_h_labels', False):

                    if idx.item() == len(edit_logits[0]) - 1:
                        done = True
                        break

                    elif idx.item() < max_bond_idx:

                        if self.config['bs_outdim'] > 1:
                            bond_logits = edit_logits[0][:products.GetNumBonds() * len(BOND_FLOATS)]
                            bond_logits = bond_logits.reshape(products.GetNumBonds(), len(BOND_FLOATS))
                            idx_tensor = torch.where(bond_logits == val)

                            idx_tensor = [indices[-1] for indices in idx_tensor]

                            bond_idx, bo_idx = idx_tensor[0].item(), idx_tensor[1].item()
                            a1 = products.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                            a2 = products.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                            a1, a2 = sorted([a1, a2])
                            bo = products.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()
                            new_bo = BOND_FLOATS[bo_idx]

                            edit = f"{a1}:{a2}:{bo}:{new_bo}"

                        elif self.config['bs_outdim'] == 1:
                            bond_idx = idx.item()
                            a1 = products.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
                            a2 = products.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

                            a1, a2 = sorted([a1, a2])
                            bo = products.GetBondWithIdx(bond_idx).GetBondTypeAsDouble()

                            edit = f"{a1}:{a2}:{bo}:{0.0}"

                        else:
                            pass

                    else:
                        h_logits = edit_logits[0][max_bond_idx:-1]
                        assert len(h_logits) == products.GetNumAtoms()

                        atom_idx = idx.item() - max_bond_idx
                        a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()

                        edit = f"{a1}:{0}:{1.0}:{0.0}"

                else:
                    raise ValueError("without h-labels not supported.")

                try:
                    products = apply_edits_to_mol(mol=Chem.Mol(products), edits=[edit])
                    prod_graph = RxnElement(mol=Chem.Mol(products), rxn_class=rxn_class)

                    prod_tensors, prod_scopes = pack_graph_feats([prod_graph],
                                                                  directed=directed, return_graphs=False,
                                                                  use_rxn_class=use_rxn_class)
                    prod_tensors = self.to_device(prod_tensors)
                    edits.append(edit)
                    steps += 1

                except:
                    steps += 1
                    continue

        edits = list(set(edits))

        return edits

    def eval_step(self, prod_smi_batch: List[str],
                  core_edits_batch: List[str], rxn_classes: List[int] = None,
                  **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Eval step of the model.

        Parameters
        ----------
        prod_smi_batch: List[str],
            List of product smiles
        core_edits_batch: List[List]:
            List of edits for each element in batch
        rxn_classes: List[int] = None,
            List of rxn classes for product in batch
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
