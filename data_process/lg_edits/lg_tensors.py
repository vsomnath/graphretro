import torch
from rdkit import Chem
import argparse
import joblib
import os
import sys

from seq_graph_retro.molgraph import RxnElement
from seq_graph_retro.data.collate_fns import pack_graph_feats
from seq_graph_retro.utils import str2bool

DATA_DIR = "./datasets/uspto-50k"
INFO_FILE = "uspto_50k.info.kekulized"
NUM_SHARDS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory to parse from.")
    parser.add_argument('--mode', default='train')
    parser.add_argument("--mpnn", default='graph_feat')
    parser.add_argument("--use_h_labels", type=str2bool, default=True, help='Whether to use h-labels')
    parser.add_argument("--use_rxn_class", type=str2bool, default=False, help='Whether to use reaction class.')

    args = parser.parse_args()

    if args.mode == 'dummy':
        lg_mols_file = os.path.join(args.data_dir, 'dummy')
    else:
        lg_mols_file = os.path.join(args.data_dir, 'train')

    if args.use_h_labels:
        lg_mols_file += "/h_labels/lg_mols.file"
    else:
        lg_mols_file += "/without_h_labels/lg_mols.file"
    lg_mols = joblib.load(lg_mols_file)

    graphs = []
    for idx, mol in enumerate(lg_mols):
        graphs.append(RxnElement(mol=Chem.Mol(mol), rxn_class=0))

    print(len(graphs))
    sys.stdout.flush()

    if args.mpnn == 'graph_feat':
        directed = True
    elif args.mpnn == 'wln':
        directed = False

    lg_inputs = pack_graph_feats(graphs, directed=directed, use_rxn_class=args.use_rxn_class)
    if args.use_h_labels:
        save_dir = os.path.join(args.data_dir, f"{args.mode}", "h_labels")
    else:
        save_dir = os.path.join(args.data_dir, f"{args.mode}", "without_h_labels")

    if args.use_rxn_class:
        save_dir = os.path.join(save_dir, "with_rxn")
    else:
        save_dir = os.path.join(save_dir, "without_rxn")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(lg_inputs, os.path.join(save_dir, f"lg_inputs.pt"))
    print("Save complete.")
    sys.stdout.flush()
