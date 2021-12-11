import numpy as np
from rdkit import Chem
import pandas as pd
import argparse
import joblib
import os
import sys
import copy
from typing import List, Any

from seq_graph_retro.utils.parse import get_reaction_info, extract_leaving_groups
from seq_graph_retro.molgraph import MultiElement
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.utils import str2bool

DATA_DIR = "./datasets/uspto-50k"


def parse_info(rxns: List, rxn_classes: List, args: Any, mode: str = 'train') -> None:
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
    info_all = []
    mol_list = []
    counter = []
    if args.use_h_labels:
        save_dir = os.path.join(args.data_dir, f"{mode}", "h_labels")
    else:
        save_dir = os.path.join(args.data_dir, f"{mode}", "without_h_labels")

    os.makedirs(save_dir, exist_ok=True)

    for idx, rxn_smi in enumerate(rxns):
        try:
            reaction_info = get_reaction_info(rxn_smi, kekulize=args.kekulize,
                                              use_h_labels=args.use_h_labels,
                                              rxn_class=int(rxn_classes[idx]))
        except:
            print(f"Failed to extract reaction info. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

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
        counter.append(len(reaction_info.core_edits))

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        frag_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(fragments)).mols)
        reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)
        mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))
        info_all.append(reaction_info)

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(rxns)} {mode} reactions processed.")
            sys.stdout.flush()

    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    info_file = os.path.join(save_dir, args.save_file)
    if args.kekulize:
        info_file += ".kekulized"

    n_shards = 5
    indices_shards = np.array_split(np.arange(len(info_all)), n_shards)

    for shard_num, indices_per_shard in enumerate(indices_shards):
        info_shard = []
        frag_shard = []

        for index in indices_per_shard:
            info_shard.append(info_all[index])

        info_file_shard = info_file + f"-shard-{shard_num}"
        joblib.dump(info_shard, info_file_shard, compress=3)

    print("Extracting leaving groups.")
    lg_dict, lg_groups, lg_mols = extract_leaving_groups(mol_list)

    print("Leaving groups extracted...")
    print(f"{mode}: {len(lg_groups)}, {len(info_all)}")
    sys.stdout.flush()

    if mode == 'train' or mode == 'dummy':
        joblib.dump(lg_dict, os.path.join(save_dir, "lg_vocab.txt"))
        joblib.dump(lg_mols, os.path.join(save_dir, "lg_mols.file"))
        print(lg_dict)

    from collections import Counter
    print(Counter(counter))
    joblib.dump(lg_groups, os.path.join(save_dir, "lg_groups.txt"))
    joblib.dump(lg_mols, os.path.join(save_dir, "lg_mols.file"))

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to parse from.")
    parser.add_argument("--save_file", default="uspto_50k.info", help='Base filename to save')
    parser.add_argument('--mode', required=True, help="Type of dataset being prepared.")
    parser.add_argument("--print_every", default=1000, type=int, help="Print during parsing.")
    parser.add_argument("--kekulize", type=str2bool, default=True, help='Whether to kekulize mols during training')
    parser.add_argument("--use_h_labels", type=str2bool, default=True, help='Whether to use h-labels')
    args = parser.parse_args()

    rxn_key = "reactants>reagents>production"
    filename = f"canonicalized_{args.mode}.csv"
    df = pd.read_csv(os.path.join(args.data_dir, filename))
    parse_info(rxns=df[rxn_key], rxn_classes=df['class'], args=args, mode=args.mode)

if __name__ == "__main__":
    main()
