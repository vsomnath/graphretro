import numpy as np
import pandas as pd
import torch
import os
import argparse
import tqdm
import wandb
import yaml

from seq_graph_retro.utils.parse import get_reaction_info
from seq_graph_retro.models import SingleEdit, MultiEdit
from seq_graph_retro.search import EditSearch

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
MODELS = {'SingleEdit': SingleEdit, "single_edit": SingleEdit,
        'MultiEdit': MultiEdit, "multi_edit": MultiEdit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
    parser.add_argument("--exp_dir", default=EXP_DIR, help="Experiments directory.")
    parser.add_argument("--test_file", default=DEFAULT_TEST_FILE, help="Test file")
    parser.add_argument("--edits_exp", default="SingleEdit_21-03-2020--20-33-05",
                        help="Name of edit prediction experiment.")
    parser.add_argument("--edits_step", default=None,
                        help="Checkpoint to load for the edits experiment.")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width")

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)

    edits_step = args.edits_step
    if edits_step is None:
        edits_step = "best_model"

    if "run" in args.edits_exp:
        # This addition because some of the new experiments were run using wandb
        edits_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.edits_exp, "files", edits_step + ".pt"), map_location=DEVICE)
        with open(f"{args.exp_dir}/wandb/{args.edits_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        edits_loaded = torch.load(os.path.join(args.exp_dir, args.edits_exp,
                                  "checkpoints", edits_step + ".pt"),
                                  map_location=DEVICE)
        model_name = args.edits_exp.split("_")[0]

    model_class = MODELS.get(model_name)
    config = edits_loaded["saveables"]

    em = model_class(**config, device=DEVICE)
    em.load_state_dict(edits_loaded['state'])
    em.to(DEVICE)
    em.eval()

    toggles = config['toggles']

    if model_name == 'single_edit' or model_name == "SingleEdit":
        beam_model = EditSearch(model=em, beam_width=args.beam_width, max_edits=1)
    else:
        beam_model = EditSearch(model=em, beam_width=args.beam_width, max_edits=6)

    pbar = tqdm.tqdm(list(range(len(test_df))))
    n_matched = np.zeros(args.beam_width)

    for idx in pbar:
        rxn_smi = test_df.loc[idx, 'reactants>reagents>production']
        r, p = rxn_smi.split(">>")
        rxn_class = test_df.loc[idx, 'class']

        if rxn_class != 'UNK':
            rxn_class = int(rxn_class)
        try:
            info = get_reaction_info(rxn_smi=rxn_smi,
                                 kekulize=True, use_h_labels=True,
                                 rxn_class=rxn_class)
            true_edit = info.core_edits

            if toggles.get("use_rxn_class", False):
                top_k_nodes = beam_model.run_edit_step(p, max_steps=6, rxn_class=rxn_class)
            else:
                top_k_nodes = beam_model.run_edit_step(p, max_steps=6)

            beam_matched = False
            for beam_idx, node in enumerate(top_k_nodes):
                edit = node.edit
                if not isinstance(edit, list):
                    edit = [edit]

                if set(edit) == set(true_edit) and not beam_matched:
                    n_matched[beam_idx] += 1
                    beam_matched = True

            msg = 'average score'
            for beam_idx in [1, 2, 3, 5]:
                match_perc = np.sum(n_matched[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.4f' % (beam_idx, match_perc)
            pbar.set_description(msg)
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    main()
