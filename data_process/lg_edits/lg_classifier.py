import torch
from rdkit import Chem
import argparse
import joblib
import os
import sys
import copy
import random
from typing import List, Any, Tuple

from seq_graph_retro.utils.parse import extract_leaving_groups, map_reac_and_frag
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.molgraph import RxnElement, MultiElement
from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.data.collate_fns import pack_graph_feats, prepare_lg_labels
from seq_graph_retro.utils import str2bool

DATA_DIR = "./datasets/uspto-50k"
INFO_FILE = "uspto_50k.info.kekulized"
NUM_SHARDS = 5

def process_batch(prod_batch: List[RxnElement],
                  frag_batch: List[MultiElement],
                  mol_list: Tuple[Chem.Mol], args: Any) -> Tuple[Tuple[torch.Tensor]]:
    """Process batch of input graphs.

    Parameters
    ----------
    prod_batch: List[RxnElement],
        Product graphs in batch
    frag_batch: List[MultiElement],
        Fragment graphs in batch
    mol_List: Tuple[Chem.Mol],
        A tuple of (product, reactant, fragment) mols to extract leaving groups
    args: Any,
        Command line arguments
    """
    assert len(frag_batch) == len(mol_list)

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

    if args.mpnn == 'graph_feat':
        directed = True
    elif args.mpnn == 'wln':
        directed = False

    prod_inputs = pack_graph_feats(prod_batch, directed=directed, use_rxn_class=args.use_rxn_class)
    frag_inputs = pack_graph_feats(frag_batch, directed=directed, use_rxn_class=args.use_rxn_class)
    lg_labels, lengths = prepare_lg_labels(lg_vocab, lg_groups)

    return prod_inputs, frag_inputs, lg_labels, lengths

def parse_frags_forward(args: Any, mode: str = 'train') -> None:
    """Parse Fragments using same order as reactants.

    Parameters
    ----------
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

    frag_graphs = []
    prod_graphs = []
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
        save_dir = os.path.join(save_dir, "with_rxn", "lg_classifier")
    else:
        save_dir = os.path.join(save_dir, "without_rxn", "lg_classifier")

    save_dir = os.path.join(save_dir, args.mpnn)
    os.makedirs(save_dir, exist_ok=True)
    num_batches = 0

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

        prod_graph = RxnElement(mol=Chem.Mol(products), rxn_class=reaction_info.rxn_class)
        tmp_frags = MultiElement(mol=Chem.Mol(fragments))
        tmp_reac = MultiElement(mol=Chem.Mol(reactants))

        frag_mols = copy.deepcopy(tmp_frags.mols)
        reac_mols = copy.deepcopy(tmp_reac.mols)

        reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)

        # Shuffling is introduced here to negate the effects that the
        # atom-mapping might bring on the order in which synthons are processed.
        shuffled_order = list(range(len(reac_mols)))
        #random.shuffle(shuffled_order)
        reac_mols = [reac_mols[idx] for idx in shuffled_order]
        frag_mols = [frag_mols[idx] for idx in shuffled_order]

        reac_aligned = Chem.Mol()
        frag_aligned = Chem.Mol()

        # Combine the shuffled mols into a single mol
        for reac_mol, frag_mol in zip(*(reac_mols, frag_mols)):
            reac_aligned = Chem.CombineMols(reac_aligned, reac_mol)
            frag_aligned = Chem.CombineMols(frag_aligned, frag_mol)

        frag_graph = MultiElement(mol=Chem.Mol(frag_aligned), rxn_class=reaction_info.rxn_class)

        prod_graphs.append(prod_graph)
        frag_graphs.append(frag_graph)
        mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

        assert len(frag_graphs) == len(mol_list) == len(prod_graphs)
        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            batch_tensors = process_batch(prod_graphs, frag_graphs, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            mol_list = []
            prod_graphs = []
            frag_graphs = []


    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    if len(frag_graphs) != 0:
        assert len(frag_graphs) == len(mol_list) == len(prod_graphs)
        batch_tensors = process_batch(prod_graphs, frag_graphs, mol_list, args)
        torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

        num_batches += 1
        mol_list = []
        prod_graphs = []
        frag_graphs = []

    return num_batches

def parse_frags_reverse(args: Any, mode: str = 'train', num_batches: int = None) -> None:
    """Parse Fragments using reverse order as reactants.

    Parameters
    ----------
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

    frag_graphs = []

    prod_graphs = []
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
        save_dir = os.path.join(save_dir, "with_rxn", "lg_classifier")
    else:
        save_dir = os.path.join(save_dir, "without_rxn", "lg_classifier")

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

        prod_graph = RxnElement(mol=Chem.Mol(products), rxn_class=reaction_info.rxn_class)
        tmp_frags = MultiElement(mol=Chem.Mol(fragments))
        tmp_reac = MultiElement(mol=Chem.Mol(reactants))

        frag_mols = tmp_frags.mols
        reac_mols = tmp_reac.mols

        reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)
        reac_mols_rev = copy.deepcopy(reac_mols[::-1])
        frag_mols_rev = copy.deepcopy(frag_mols[::-1])

        fragments_rev = Chem.Mol()
        reactants_rev = Chem.Mol()

        for reac_mol, frag_mol in zip(*(reac_mols_rev, frag_mols_rev)):
            fragments_rev = Chem.CombineMols(fragments_rev, Chem.Mol(frag_mol))
            reactants_rev = Chem.CombineMols(reactants_rev, Chem.Mol(reac_mol))

        frag_graph = MultiElement(mol=Chem.Mol(fragments_rev), rxn_class=reaction_info.rxn_class)
        reac_graph = MultiElement(mol=Chem.Mol(reactants_rev), rxn_class=reaction_info.rxn_class)

        prod_graphs.append(prod_graph)
        frag_graphs.append(frag_graph)
        mol_list.append((products, copy.deepcopy(reac_mols_rev), copy.deepcopy(frag_mols_rev)))

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

        assert len(frag_graphs) == len(mol_list) == len(prod_graphs)
        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            batch_tensors = process_batch(prod_graphs, frag_graphs, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            mol_list = []
            prod_graphs = []
            frag_graphs = []


    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    if len(frag_graphs) != 0:
        assert len(frag_graphs) == len(mol_list) == len(prod_graphs)
        batch_tensors = process_batch(prod_graphs, frag_graphs, mol_list, args)
        torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

        num_batches += 1
        mol_list = []
        prod_graphs = []
        frag_graphs = []


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to parse from.")
    parser.add_argument("--info_file", default=INFO_FILE, help='File with the information.')
    parser.add_argument("--print_every", default=1000, type=int, help="Print during parsing.")
    parser.add_argument('--mode', default='train')
    parser.add_argument("--mpnn", default='graph_feat')
    parser.add_argument("--use_h_labels", type=str2bool, default=True, help='Whether to use h-labels')
    parser.add_argument("--use_rxn_class", type=str2bool, default=False, help='Whether to use reaction-class')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size to use.')
    parser.add_argument("--augment", action='store_true', help="Whether to augment")

    args = parser.parse_args()

    if args.augment:
        num_batches = parse_frags_forward(args=args, mode=args.mode)
        parse_frags_reverse(args=args, num_batches=num_batches, mode=args.mode)
    else:
        num_batches = parse_frags_forward(args=args, mode=args.mode)

if __name__ == "__main__":
    main()
