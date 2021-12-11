import torch
from rdkit import Chem
import argparse
import joblib
import os
import sys
import copy
from typing import Any

from seq_graph_retro.utils.parse import extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.molgraph import BondEditsRxn, RxnElement, MultiElement
from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.data.collate_fns import pack_graph_feats, prepare_lg_labels
from seq_graph_retro.utils import str2bool

DATA_DIR = "./datasets/uspto-50k"
INFO_FILE = "uspto_50k.info.kekulized"
NUM_SHARDS = 5

def check_edits(edits):
    for edit in edits:
        a1, a2, b1, b2 = edit.split(":")
        b1 = float(b1)
        if b1 == 0.0:
            return False

    return True

def process_batch_seq(edit_graphs, frag_batch, mol_list, args):
    assert len(edit_graphs) == len(frag_batch) == len(mol_list)
    lengths = torch.tensor([len(graph_seq) for graph_seq in edit_graphs], dtype=torch.long)
    max_seq_len = max([len(graph_seq) for graph_seq in edit_graphs])

    if args.mode == 'dummy':
        lg_vocab_file = os.path.join(args.data_dir, 'dummy')
    else:
        lg_vocab_file = os.path.join(args.data_dir, 'train')

    if args.use_h_labels:
        lg_vocab_file += "/h_labels/lg_vocab.txt"
    else:
        lg_vocab_file += "/without_h_labels/lg_vocab.txt"

    lg_vocab = Vocab(joblib.load(lg_vocab_file))

    seq_tensors = []
    seq_labels = []

    _, lg_groups, _ = extract_leaving_groups(mol_list)

    mol_attrs = ['prod_mol']
    if args.use_h_labels:
        label_attrs = ['edit_label', 'h_label', 'done_label']
    else:
        label_attrs = ['edit_label', 'done_label']

    seq_mask = []
    if args.mpnn == 'graph_feat':
        directed = True
    elif args.mpnn == 'wln':
        directed = False

    for idx in range(max_seq_len):
        graphs_idx = [copy.deepcopy(edit_graphs[i][min(idx, length-1)]).get_attributes(mol_attrs=mol_attrs,
                                                                label_attrs=label_attrs)
                     for i, length in enumerate(lengths)]
        mask = (idx < lengths).long()
        prod_graphs, edit_labels = list(zip(*graphs_idx))
        assert all([isinstance(graph, RxnElement) for graph in prod_graphs])

        if len(edit_labels[0]) == 1:
            edit_labels = torch.tensor(edit_labels, dtype=torch.long)
        else:
            edit_labels = [torch.tensor(edit_labels[i], dtype=torch.float) for i in range(len(edit_labels))]

        prod_tensors = pack_graph_feats(prod_graphs, directed=directed, use_rxn_class=args.use_rxn_class)
        seq_tensors.append(prod_tensors)
        seq_labels.append(edit_labels)
        seq_mask.append(mask)

    frag_tensors = pack_graph_feats(frag_batch, directed=directed, use_rxn_class=args.use_rxn_class)
    lg_labels, lengths = prepare_lg_labels(lg_vocab, lg_groups)
    seq_mask = torch.stack(seq_mask).long()
    assert seq_mask.shape[0] == max_seq_len
    assert seq_mask.shape[1] == len(mol_list)

    return seq_tensors, seq_labels, seq_mask, frag_tensors, lg_labels, lengths

def parse_bond_edits_seq(args: Any, mode: str = 'train') -> None:
    """Parse reactions.

    Parameters
    ----------
    rxns: List
        List of reaction SMILES
    args: Namespace object
        Args supplied via command line
    mode: str, default train
        Type of dataset being parsed.
    """
    if args.use_h_labels:
        base_file = os.path.join(args.data_dir, f"{mode}", "h_labels", args.info_file)
    else:
        base_file = os.path.join(args.data_dir, f"{mode}", "without_h_labels", args.info_file)

    info_shards = 5
    info_all = []
    for shard_num in range(info_shards):
        shard_file = base_file + f"-shard-{shard_num}"
        info_all.extend(joblib.load(shard_file))

    bond_edits_graphs = []
    bond_edits_frags = []
    mol_list = []

    save_dir = os.path.join(args.data_dir, f"{mode}")

    if args.use_h_labels:
        save_dir = os.path.join(save_dir, "h_labels")
    else:
        save_dir = os.path.join(save_dir, "without_h_labels")

    if args.use_rxn_class:
        save_dir = os.path.join(save_dir, "with_rxn", "bond_edits_seq")
    else:
        save_dir = os.path.join(save_dir, "without_rxn", "bond_edits_seq")
    save_dir = os.path.join(save_dir, args.mpnn)
    os.makedirs(save_dir, exist_ok=True)

    num_batches = 0

    for idx, reaction_info in enumerate(info_all):
        graph_seq = []
        rxn_smi = reaction_info.rxn_smi
        r, p = rxn_smi.split(">>")
        products = get_mol(p)

        if (products is None) or (products.GetNumAtoms() <= 1):
            print(f"Product has 0 or 1 atoms, Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        reactants = get_mol(r)

        if (reactants is None) or (reactants.GetNumAtoms() <= 1):
            print(f"Reactant has 0 or 1 atoms, Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        fragments = apply_edits_to_mol(Chem.Mol(products), reaction_info.core_edits)

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        tmp_frag = MultiElement(mol=Chem.Mol(fragments)).mols
        fragments = Chem.Mol()
        for mol in tmp_frag:
            fragments = Chem.CombineMols(fragments, mol)

        edits_accepted = check_edits(reaction_info.core_edits)
        if not edits_accepted:
            print(f"New addition edit. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        edits_applied = []
        for _, edit in enumerate(reaction_info.core_edits):
            interim_mol = apply_edits_to_mol(Chem.Mol(products), edits_applied)
            if interim_mol is None:
                print("Interim mol is None")
                break
            graph = BondEditsRxn(prod_mol=Chem.Mol(interim_mol),
                                 frag_mol=Chem.Mol(fragments),
                                 reac_mol=Chem.Mol(reactants),
                                 edits_to_apply=[edit],
                                 rxn_class=reaction_info.rxn_class)
            edits_applied.append(edit)
            graph_seq.append(graph)

        interim_mol = apply_edits_to_mol(Chem.Mol(products), edits_applied)
        if interim_mol is not None:
            graph = BondEditsRxn(prod_mol=Chem.Mol(interim_mol),
                                 frag_mol=Chem.Mol(fragments),
                                 reac_mol=Chem.Mol(reactants),
                                 edits_to_apply=[],
                                 rxn_class=reaction_info.rxn_class)

            frag_graph = MultiElement(mol=Chem.Mol(fragments),
                                      rxn_class=reaction_info.rxn_class)

            frag_mols = copy.deepcopy(frag_graph.mols)
            reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)

            graph_seq.append(graph)
        else:
            continue

        if len(graph_seq) == 0:
            print(f"No valid fragment states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        bond_edits_graphs.append(graph_seq)
        bond_edits_frags.append(frag_graph)
        mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

        assert len(bond_edits_graphs) == len(bond_edits_frags) == len(mol_list)
        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            batch_tensors = process_batch_seq(bond_edits_graphs, bond_edits_frags, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            bond_edits_frags = []
            bond_edits_graphs = []
            mol_list = []

    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    batch_tensors = process_batch_seq(bond_edits_graphs, bond_edits_frags, mol_list, args)
    print("Saving..")
    torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

    num_batches += 1
    bond_edits_frags = []
    bond_edits_graphs = []
    mol_list = []

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to parse from.")
    parser.add_argument("--info_file", default=INFO_FILE, help='File with the information.')
    parser.add_argument("--print_every", default=1000, type=int, help="Print during parsing.")
    parser.add_argument('--mode', default='train')
    parser.add_argument("--mpnn", default='graph_feat')
    parser.add_argument("--use_h_labels", type=str2bool, default=True, help='Whether to use h-labels')
    parser.add_argument("--use_rxn_class", type=str2bool, default=False, help='Whether to use rxn_class')
    parser.add_argument("--batch_size", default=32, type=int, help="Number of shards")

    args = parser.parse_args()
    parse_bond_edits_seq(args=args, mode=args.mode)

if __name__ == "__main__":
    main()
