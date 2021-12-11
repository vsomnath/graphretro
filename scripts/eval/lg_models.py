import numpy as np
from rdkit import Chem
import pandas as pd
import torch
import os
import argparse
import tqdm
import yaml

from seq_graph_retro.utils.parse import get_reaction_info, extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol
from seq_graph_retro.molgraph import MultiElement
from seq_graph_retro.models import LGClassifier, LGIndEmbed
from seq_graph_retro.search import LGSearch

try:
    ROOT_DIR = os.environ["SEQ_GRAPH_RETRO"]
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "experiments")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "local_experiments")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TEST_FILE = f"{DATA_DIR}/canonicalized_test.csv"
MODELS = {'LGClassifier': LGClassifier, "lg_classifier": LGClassifier,
        "LGIndEmbed": LGIndEmbed, "lg_ind": LGIndEmbed, "LGIndEmbedClassifier": LGIndEmbed}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
    parser.add_argument("--exp_dir", default=EXP_DIR, help="Experiments directory.")
    parser.add_argument("--test_file", default=DEFAULT_TEST_FILE, help="Test file")
    parser.add_argument("--lg_exp", default="LGClassifier_02-04-2020--02-06-17",
                        help="Name of synthon completion experiment")
    parser.add_argument("--lg_step", default=None,
                        help="Checkpoint from synthon completion experiment")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)

    lg_step = args.lg_step
    if lg_step is None:
        lg_step = "best_model"

    if "run" in args.lg_exp:
        # This addition because some of the new experiments were run using wandb
        lg_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.lg_exp, "files", lg_step + ".pt"), map_location=DEVICE)
        with open(f"{args.exp_dir}/wandb/{args.lg_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        lg_loaded = torch.load(os.path.join(args.exp_dir, args.lg_exp,
                               "checkpoints", lg_step + ".pt"),
                                map_location=DEVICE)
        model_name = args.lg_exp.split("_")[0]

    model_class = MODELS.get(model_name)
    config = lg_loaded["saveables"]
    toggles = config['toggles']

    if 'tensor_file' in config:
        if not os.path.isfile(config['tensor_file']):
            if not toggles.get("use_rxn_class", False):
                tensor_file = os.path.join(args.data_dir, "train/h_labels/without_rxn/lg_inputs.pt")
            else:
                tensor_file = os.path.join(args.data_dir, "train/h_labels/with_rxn/lg_inputs.pt")
            config['tensor_file'] = tensor_file

    lg_model = model_class(**config, device=DEVICE)
    lg_model.load_state_dict(lg_loaded['state'])
    lg_model.to(DEVICE)
    lg_model.eval()

    n_matched = np.zeros(args.beam_width)
    beam_model = LGSearch(model=lg_model, beam_width=args.beam_width, max_edits=1)
    pbar = tqdm.tqdm(list(range(len(test_df))))

    for idx in pbar:
        rxn_smi = test_df.loc[idx, 'reactants>reagents>production']
        r, p = rxn_smi.split(">>")
        rxn_class = test_df.loc[idx, 'class']

        if rxn_class != 'UNK':
            rxn_class = int(rxn_class)

        info = get_reaction_info(rxn_smi=rxn_smi,
                                 kekulize=True, use_h_labels=True,
                                 rxn_class=rxn_class)

        reactants = Chem.MolFromSmiles(r)
        products = Chem.MolFromSmiles(p)
        fragments = apply_edits_to_mol(Chem.Mol(products), info.core_edits)

        frag_mols = MultiElement(Chem.Mol(fragments)).mols
        reac_mols = MultiElement(Chem.Mol(reactants)).mols

        _, labels, _ = extract_leaving_groups([(products, reac_mols, frag_mols)])
        assert len(labels) == 1
        lg_group = labels[0]

        if toggles.get("use_rxn_class", False):
            top_k_nodes = beam_model.run_search(p, edits=info.core_edits, max_steps=6, rxn_class=rxn_class)
        else:
            top_k_nodes = beam_model.run_search(p, edits=info.core_edits, max_steps=6)

        beam_matched = False
        for beam_idx, node in enumerate(top_k_nodes):
            pred_labels = node.lg_groups
            if pred_labels == lg_group and not beam_matched:
                n_matched[beam_idx] += 1
                beam_matched = True

        msg = 'average score'
        for beam_idx in [1, 2, 3, 5]:
            match_perc = np.sum(n_matched[:beam_idx]) / (idx + 1)
            msg += ', t%d: %.4f' % (beam_idx, match_perc)

        pbar.set_description(msg)

if __name__ == "__main__":
    main()
