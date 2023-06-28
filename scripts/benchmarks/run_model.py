import torch
import os
import argparse
import json
from datetime import datetime as dt
import sys
from rdkit import RDLogger

from seq_graph_retro.molgraph.mol_features import ATOM_FDIM
from seq_graph_retro.molgraph.mol_features import BOND_FDIM, BINARY_FDIM
from seq_graph_retro.models.model_builder import build_model, MODEL_ATTRS
from seq_graph_retro.models import Trainer
from seq_graph_retro.utils import str2bool
import wandb
import yaml
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
    ROOT_DIR = os.environ["SEQ_GRAPH_RETRO"]
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    out_dir = os.path.join(ROOT_DIR, "experiments")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    out_dir = os.path.join(ROOT_DIR, "local_experiments")

INFO_FILE = "uspto_50k.info.kekulized"
LABELS_FILE = "lg_groups.txt"
VOCAB_FILE = "lg_vocab.txt"

NUM_SHARDS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_dir(model_name):
    MODEL_DIRS = {
        'single_edit': 'bond_edits',
        'multi_edit': 'bond_edits_seq',
        'single_shared': 'bond_edits',
        'lg_classifier': 'lg_classifier',
        'lg_ind': 'lg_classifier'
    }

    return MODEL_DIRS.get(model_name)


def run_model(config):
    print(config)
    if config.get('use_doubles', False):
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if config.get('use_augment', False):
        aug_suffix = "_aug"
    else:
        aug_suffix = ""

    model = build_model(config, device=DEVICE)
    print(f"Converting model to device: {DEVICE}")
    sys.stdout.flush()
    model.to(DEVICE)

    if config.get('restore_enc_from', None):
        ckpt_id = config.get("enc_ckpt", "best_model")
        loaded = torch.load(os.path.join(config['out_dir'], "wandb", config['restore_enc_from'], "files",
                                         f"{ckpt_id}.pt"), map_location=DEVICE)
        if 'saveables' in loaded:
            loaded_enc_name = loaded['saveables']['encoder_name']
            msg = "Encoder name of pretrained encoder and current encoder must match"
            assert loaded_enc_name == model.encoder_name, msg

        state = loaded['state']
        enc_keys = [key for key in state.keys() if 'encoder' in key]
        enc_dict = {key: state[key] for key in enc_keys}

        model_dict = model.state_dict()
        model_dict.update(enc_dict)
        for key in enc_keys:
            assert torch.sum(torch.eq(model_dict[key], enc_dict[key]))

        print("Loading from pretrained encoder.")
        sys.stdout.flush()
        model.load_state_dict(model_dict)

    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10**6, "M")
    print()
    sys.stdout.flush()

    print(f"Device used: {DEVICE}")
    sys.stdout.flush()

    _, train_dataset_class, eval_dataset_class, use_labels = MODEL_ATTRS.get(config['model'])
    model_dir_name = get_model_dir(config['model'])

    if config.get('use_h_labels', True):
        train_dir = os.path.join(config['data_dir'], "train" + aug_suffix, "h_labels")
        eval_dir = os.path.join(config['data_dir'], "eval", "h_labels")
    else:
        train_dir = os.path.join(config['data_dir'], "train" + aug_suffix, "without_h_labels")
        eval_dir = os.path.join(config['data_dir'], "eval", "without_h_labels")

    if config.get('use_rxn_class', False):
        train_dir = os.path.join(train_dir, "with_rxn", model_dir_name)
    else:
        train_dir = os.path.join(train_dir, "without_rxn", model_dir_name)

    train_dataset = train_dataset_class(data_dir=train_dir,
                                        mpnn=config['mpnn'])

    if eval_dataset_class is not None:
        eval_dataset = eval_dataset_class(data_dir=eval_dir,
                                              data_file=config['info_file'],
                                              labels_file=config['labels_file'] if use_labels else None,
                                              use_rxn_class=config.get('use_rxn_class', False),
                                              num_shards=config['num_shards'])

    train_data = train_dataset.create_loader(batch_size=1, shuffle=True,
                                             num_workers=config['num_workers'])

    if eval_dataset_class is None:
        eval_data = None
    else:
        eval_data = eval_dataset.create_loader(batch_size=1,
                                               num_workers=config['num_workers'])

    date_and_time = dt.now().strftime("%d-%m-%Y--%H-%M-%S")

    trainer = Trainer(model=model, print_every=config['print_every'],
                      eval_every=config['eval_every'])
    trainer.build_optimizer(learning_rate=config['lr'], finetune_encoder=False)
    trainer.build_scheduler(type=config['scheduler_type'], anneal_rate=config['anneal_rate'],
                            patience=config['patience'], thresh=config['metric_thresh'])
    trainer.train_epochs(train_data, eval_data, config['epochs'],
                         **{"accum_every": config.get('accum_every', None),
                            "clip_norm": config['clip_norm']})

def main(args):
    # initialize wandb
    wandb.init(project='seq_graph_retro', dir=args.out_dir,
               config=args.config_file)
    config = wandb.config
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        config[key] = value

    run_model(config)


def sweep(args):
    # load config
    with open(args.config_file) as file:
        default_config = yaml.load(file, Loader=yaml.FullLoader)

    loaded_config = {}
    for key in default_config:
        loaded_config[key] = default_config[key]['value']
    
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        loaded_config[key] = value

    # init wandb
    wandb.init(allow_val_change=True, dir=args.out_dir)

    # update wandb config
    wandb.config.update(loaded_config)
    config = wandb.config

    # start run
    run_model(config)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
    parser.add_argument("--out_dir", default=out_dir, help="Experiments directory")
    parser.add_argument("--info_file", default=INFO_FILE,
                        help="File containing info. Used only for validation")
    parser.add_argument("--labels_file", default=LABELS_FILE,
                        help='File containing leaving groups. Used only for validation')
    parser.add_argument("--vocab_file", default=VOCAB_FILE,
                        help='File containing the vocabulary of leaving groups.')
    parser.add_argument("--num_shards", default=NUM_SHARDS, help="Number of shards")
    parser.add_argument("--num_workers", default=6, help="Number of workers")
    parser.add_argument("--config_file", required=True, help='File containing the configuration.')
    parser.add_argument("--sweep", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.out_dir):
       os.mkdir(args.out_dir)

    if args.sweep:
        sweep(args)
    else:
        main(args)
