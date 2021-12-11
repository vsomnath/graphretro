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
from seq_graph_retro.data.collate_fns import (pack_graph_feats, prepare_lg_labels,
        tensorize_bond_graphs)
from seq_graph_retro.utils import str2bool

DATA_DIR = "./datasets/uspto-50k"
INFO_FILE = "uspto_50k.info.kekulized"
NUM_SHARDS = 5

def process_batch(edit_graphs, mol_list, args):
    assert len(edit_graphs) == len(mol_list)

    if args.mode == 'dummy':
        lg_vocab_file = os.path.join(args.data_dir, 'dummy')
    else:
        lg_vocab_file = os.path.join(args.data_dir, 'train')

    if args.use_h_labels:
        lg_vocab_file += "/h_labels/lg_vocab.txt"
    else:
        lg_vocab_file += "/without_h_labels/lg_vocab.txt"

    lg_vocab = Vocab(joblib.load(lg_vocab_file))
    _, lg_groups, _ = extract_leaving_groups(mol_list)

    mol_attrs = ['prod_mol', 'frag_mol']
    if args.use_h_labels:
        label_attrs = ['edit_label', 'h_label']
    else:
        label_attrs = ['edit_label', 'done_label']

    attributes = [graph.get_attributes(mol_attrs=mol_attrs, label_attrs=label_attrs) for graph in edit_graphs]
    prod_batch, frag_batch, edit_labels = list(zip(*attributes))

    if len(edit_labels[0]) == 1:
        edit_labels = torch.tensor(edit_labels, dtype=torch.long)
    else:
        edit_labels = [torch.tensor(edit_labels[i], dtype=torch.float) for i in range(len(edit_labels))]

    if args.mpnn == 'graph_feat':
        directed = True
    elif args.mpnn == 'wln':
        directed = False

    prod_inputs = pack_graph_feats(prod_batch, directed=directed, use_rxn_class=args.use_rxn_class)
    frag_inputs = pack_graph_feats(frag_batch, directed=directed, use_rxn_class=args.use_rxn_class)
    lg_labels, lengths = prepare_lg_labels(lg_vocab, lg_groups)

    if args.parse_bond_graph:
        bond_graph_inputs = tensorize_bond_graphs(prod_batch, directed=directed, use_rxn_class=args.use_rxn_class)
        return prod_inputs, edit_labels, frag_inputs, lg_labels, lengths, bond_graph_inputs
    return prod_inputs, edit_labels, frag_inputs, lg_labels, lengths, None

def parse_bond_edits_forward(args: Any, mode: str = 'train') -> None:
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

    info_all = []
    for shard_num in range(5):
        shard_file = base_file + f"-shard-{shard_num}"
        info_all.extend(joblib.load(shard_file))

    bond_edits_graphs = []
    mol_list = []

    if args.augment:
        save_dir = os.path.join(args.data_dir, f"{mode}_aug")
    else:
        save_dir = os.path.join(args.data_dir, f"{mode}")

    if args.use_h_labels:
        save_dir = os.path.join(save_dir, "h_labels")
    else:
        save_dir = os.path.join(save_dir, "without_h_labels")

    if args.use_rxn_class:
        save_dir = os.path.join(save_dir, "with_rxn", "bond_edits")
    else:
        save_dir = os.path.join(save_dir, "without_rxn", "bond_edits")

    save_dir = os.path.join(save_dir, args.mpnn)
    os.makedirs(save_dir, exist_ok=True)

    num_batches = 0
    total_examples = 0

    for idx, reaction_info in enumerate(info_all):
        rxn_smi = reaction_info.rxn_smi
        r, p = rxn_smi.split(">>")
        products = get_mol(p)

        assert len(bond_edits_graphs) == len(mol_list)
        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            print(f"Saving after {total_examples}")
            sys.stdout.flush()
            batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            mol_list = []
            bond_edits_graphs = []

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

        if (fragments is None) or (fragments.GetNumAtoms() <=1):
            print(f"Fragments are invalid. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        tmp_frag = MultiElement(mol=Chem.Mol(fragments)).mols
        fragments = Chem.Mol()
        for mol in tmp_frag:
            fragments = Chem.CombineMols(fragments, mol)

        if len(reaction_info.core_edits) == 1:
            edit = reaction_info.core_edits[0]
            a1, a2, b1, b2 = edit.split(":")

            if float(b1) and float(b2) >= 0:
                bond_edits_graph = BondEditsRxn(prod_mol=Chem.Mol(products),
                                                frag_mol=Chem.Mol(fragments),
                                                reac_mol=Chem.Mol(reactants),
                                                edits_to_apply=[edit],
                                                rxn_class=reaction_info.rxn_class)
                frag_graph = MultiElement(mol=Chem.Mol(fragments))

                frag_mols = copy.deepcopy(frag_graph.mols)
                reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)

                bond_edits_graphs.append(bond_edits_graph)
                mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))
                total_examples += 1

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    assert len(bond_edits_graphs) == len(mol_list)
    batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
    torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

    num_batches += 1
    mol_list = []
    bond_edits_graphs = []

    return num_batches

def parse_bond_edits_reverse(args: Any, mode: str = 'train', num_batches: int = None) -> None:
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

    info_all = []
    for shard_num in range(5):
        shard_file = base_file + f"-shard-{shard_num}"
        info_all.extend(joblib.load(shard_file))

    bond_edits_graphs = []
    mol_list = []

    if args.augment:
        save_dir = os.path.join(args.data_dir, f"{mode}_aug")
    else:
        save_dir = os.path.join(args.data_dir, f"{mode}")

    if args.use_h_labels:
        save_dir = os.path.join(save_dir, "h_labels")
    else:
        save_dir = os.path.join(save_dir, "without_h_labels")

    if args.use_rxn_class:
        save_dir = os.path.join(save_dir, "with_rxn", "bond_edits")
    else:
        save_dir = os.path.join(save_dir, "without_rxn", "bond_edits")
    os.makedirs(save_dir, exist_ok=True)

    for idx, reaction_info in enumerate(info_all):
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

        if (fragments is None) or (fragments.GetNumAtoms() <=1):
            print(f"Fragments are invalid. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        if len(Chem.rdmolops.GetMolFrags(fragments)) == 1:
            continue

        tmp_frag = MultiElement(mol=Chem.Mol(fragments)).mols
        fragments = Chem.Mol()
        for mol in tmp_frag:
            fragments = Chem.CombineMols(fragments, mol)

        if len(reaction_info.core_edits) == 1:
            edit = reaction_info.core_edits[0]
            a1, a2, b1, b2 = edit.split(":")

            if float(b1) and float(b2) >= 0 and int(a2) != 0:

                frag_mols = MultiElement(mol=fragments).mols
                reac_mols = MultiElement(mol=reactants).mols

                reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)
                reac_mols_rev = copy.deepcopy(reac_mols[::-1])
                frag_mols_rev = copy.deepcopy(frag_mols[::-1])

                reactants_rev = Chem.Mol()
                for mol in reac_mols_rev:
                    reactants_rev = Chem.CombineMols(reactants_rev, Chem.Mol(mol))

                fragments_rev = Chem.Mol()
                for mol in frag_mols_rev:
                    fragments_rev = Chem.CombineMols(fragments_rev, Chem.Mol(mol))

                bond_edits_graph = BondEditsRxn(prod_mol=Chem.Mol(products),
                                                frag_mol=Chem.Mol(fragments_rev),
                                                reac_mol=Chem.Mol(reactants_rev),
                                                edits_to_apply=[edit],
                                                rxn_class=reaction_info.rxn_class)
                bond_edits_graphs.append(bond_edits_graph)
                mol_list.append((products, copy.deepcopy(reac_mols_rev), copy.deepcopy(frag_mols_rev)))

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

        assert len(bond_edits_graphs) == len(mol_list)
        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            mol_list = []
            bond_edits_graphs = []

    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    assert len(bond_edits_graphs) == len(mol_list)
    batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
    torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

    num_batches += 1
    mol_list = []
    bond_edits_graphs = []

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to parse from.")
    parser.add_argument("--info_file", default=INFO_FILE, help='File with the information.')
    parser.add_argument("--print_every", default=1000, type=int, help="Print during parsing.")
    parser.add_argument('--mode', default='train')
    parser.add_argument("--mpnn", default='graph_feat')
    parser.add_argument("--use_h_labels", type=str2bool, default=True, help='Whether to use h-labels')
    parser.add_argument("--use_rxn_class", type=str2bool, default=False, help='Whether to use rxn_class')
    parser.add_argument("--parse_bond_graph", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size to use.')
    parser.add_argument("--augment", type=str2bool, default=False, help="Whether to augment")

    args = parser.parse_args()

    if args.augment:
        num_batches = parse_bond_edits_forward(args=args, mode=args.mode)
        parse_bond_edits_reverse(args=args, num_batches=num_batches, mode=args.mode)
    else:
        num_batches = parse_bond_edits_forward(args=args, mode=args.mode)

if __name__ == "__main__":
    main()
