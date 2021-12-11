import os
import joblib

from seq_graph_retro.models import (SingleEdit, MultiEdit, SingleEditShared,
    LGClassifier, LGIndEmbed)
from seq_graph_retro.molgraph.vocab import Vocab

from seq_graph_retro.data import (SingleEditDataset, MultiEditDataset,
        SingleEditSharedDataset, LGClassifierDataset, LGEvalDataset, EditsEvalDataset,
        SharedEvalDataset)

from seq_graph_retro.molgraph.mol_features import (ATOM_FDIM, BOND_FDIM, BOND_TYPES,
                RXN_CLASSES, BOND_FLOATS, PATTERN_DIM, BINARY_FDIM)
from seq_graph_retro.molgraph.vocab import ATOM_LIST

MODEL_ATTRS = {
    'single_edit': (SingleEdit, SingleEditDataset, EditsEvalDataset, False),
    'multi_edit': (MultiEdit, MultiEditDataset, EditsEvalDataset, False),
    'single_shared': (SingleEditShared, SingleEditSharedDataset, SharedEvalDataset, True),
    'lg_classifier': (LGClassifier, LGClassifierDataset, LGEvalDataset, True),
    'lg_ind': (LGIndEmbed, LGClassifierDataset, LGEvalDataset, True)
}

def build_edits_config(loaded_config):
    model_config = {}
    config = {}
    if loaded_config.get('use_rxn_class', False):
        config['n_atom_feat'] = ATOM_FDIM + len(RXN_CLASSES)
    else:
        config['n_atom_feat'] = ATOM_FDIM
    config['n_bond_feat'] = BOND_FDIM
    config['n_bin_feat'] = BINARY_FDIM
    config['rnn_type'] = loaded_config['rnn_type']
    config['mpn_size'] = loaded_config['mpn_size']
    config['mlp_size'] = loaded_config['mlp_size']
    config['depth'] = loaded_config['depth']
    config['bias'] = False
    config['edit_loss'] = loaded_config['loss_type']
    if 'n_mt_blocks' in loaded_config:
        config['n_mt_blocks'] = loaded_config['n_mt_blocks']

    if loaded_config['edits_type'] == 'bond_edits':
        bs_outdim = len(BOND_FLOATS)
    elif loaded_config['edits_type'] == 'bond_disconn':
        bs_outdim = 1
    else:
        raise ValueError()

    config['bs_outdim'] = bs_outdim
    if loaded_config.get("propagate_logits", False):
        if loaded_config.get('use_rxn_class', False):
            config['bond_label_feat'] = ATOM_FDIM + 1 + 2 * (BOND_FDIM-1) + len(RXN_CLASSES)
        else:
            config['bond_label_feat'] = ATOM_FDIM + 1 + 2 * (BOND_FDIM-1)
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['dropout_mpn'] = loaded_config['dropout_mpn']
    config['pos_weight'] = loaded_config['pos_weight']

    toggles = {}
    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_rxn_class'] = loaded_config.get('use_rxn_class', False)
    toggles['use_h_labels'] = loaded_config.get('use_h_labels', True)
    toggles['use_prod'] = loaded_config.get('use_prod_edits', False)
    toggles['propagate_logits'] = loaded_config.get('propagate_logits', False)
    toggles['use_res'] = loaded_config.get('use_res', False)

    if 'n_heads' in loaded_config:
        config['n_heads'] = loaded_config['n_heads']

    model_config['config'] = config
    model_config['toggles'] = toggles
    return model_config

def build_shared_edit_config(loaded_config):
    model_config = {}
    config = {}
    if loaded_config.get('use_rxn_class', False):
        config['n_atom_feat'] = ATOM_FDIM + len(RXN_CLASSES)
    else:
        config['n_atom_feat'] = ATOM_FDIM
    config['n_bond_feat'] = BOND_FDIM
    config['n_bin_feat'] = loaded_config['n_bin_feat']
    config['rnn_type'] = loaded_config['rnn_type']
    config['mpn_size'] = loaded_config['mpn_size']
    config['mlp_size'] = loaded_config['mlp_size']
    config['depth'] = loaded_config['depth']
    config['bias'] = False
    config['embed_size'] = loaded_config['embed_size']
    config['edit_loss'] = loaded_config['loss_type']

    if loaded_config['edits_type'] == 'bond_edits':
        bs_outdim = len(BOND_FLOATS)
    elif loaded_config['edits_type'] == 'bond_disconn':
        bs_outdim = 1
    else:
        raise ValueError()
    config['bs_outdim'] = bs_outdim
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['dropout_mpn'] = loaded_config['dropout_mpn']

    toggles = {}
    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_prev_pred'] = loaded_config.get('use_prev_pred', True)
    toggles['use_rxn_class'] = loaded_config.get('use_rxn_class', False)
    toggles['use_h_labels'] = loaded_config.get('use_h_labels', True)
    toggles['use_prod'] = loaded_config.get('use_prod_edits', False)
    toggles['propagate_logits'] = loaded_config.get('propagate_logits', False)
    toggles['use_res'] = loaded_config.get('use_res', False)

    if 'n_heads' in loaded_config:
        config['n_heads'] = loaded_config['n_heads']

    config['embed_bias'] = False
    config['lam_edits'] = loaded_config['lam_edits']
    config['lam_lg'] = loaded_config['lam_lg']

    if loaded_config.get('use_h_labels', True):
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "h_labels")
    else:
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "without_h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "without_h_labels")

    if loaded_config.get('use_rxn_class', False):
        lg_tensor_dir = os.path.join(lg_tensor_dir, "with_rxn")
    else:
        lg_tensor_dir = os.path.join(lg_tensor_dir, "without_rxn")

    lg_tensor_file = os.path.join(lg_tensor_dir, "lg_inputs.pt")
    lg_vocab = Vocab(lg_dict)

    model_config['config'] = config
    model_config['toggles'] = toggles
    model_config['lg_vocab'] = lg_vocab
    model_config['tensor_file'] = lg_tensor_file
    return model_config

def build_lg_classifier_config(loaded_config):
    model_config = {}
    config = {}
    config['rnn_type'] = loaded_config['rnn_type']
    config['mpn_size'] = loaded_config['mpn_size']
    config['mlp_size'] = loaded_config['mlp_size']
    config['depth'] = loaded_config['depth']
    config['bias'] = False
    config['embed_size'] = loaded_config['embed_size']
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['dropout_mpn'] = loaded_config['dropout_mpn']
    if 'n_mt_blocks' in loaded_config:
        config['n_mt_blocks'] = loaded_config['n_mt_blocks']

    if loaded_config.get('use_rxn_class', False):
        config['n_atom_feat'] = ATOM_FDIM + len(RXN_CLASSES)
    else:
        config['n_atom_feat'] = ATOM_FDIM
    config['n_bond_feat'] = BOND_FDIM

    toggles = {}
    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_prev_pred'] = loaded_config.get('use_prev_pred', True)
    toggles['use_rxn_class'] = loaded_config.get('use_rxn_class', False)

    if 'n_heads' in loaded_config:
        config['n_heads'] = loaded_config['n_heads']
    config['embed_bias'] = False

    if loaded_config.get('use_h_labels', True):
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "h_labels")
    else:
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "without_h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "without_h_labels")

    if loaded_config.get('use_rxn_class', False):
        lg_tensor_dir = os.path.join(lg_tensor_dir, "with_rxn")
    else:
        lg_tensor_dir = os.path.join(lg_tensor_dir, "without_rxn")

    lg_tensor_file = os.path.join(lg_tensor_dir, "lg_inputs.pt")
    lg_vocab = Vocab(lg_dict)

    model_config['config'] = config
    model_config['toggles'] = toggles
    model_config['lg_vocab'] = lg_vocab
    model_config['tensor_file'] = lg_tensor_file
    return model_config

def build_lg_ind_config(loaded_config):
    model_config = {}
    config = {}
    config['rnn_type'] = loaded_config['rnn_type']
    config['mpn_size'] = loaded_config['mpn_size']
    config['mlp_size'] = loaded_config['mlp_size']
    config['depth'] = loaded_config['depth']
    config['bias'] = False
    config['embed_size'] = loaded_config['embed_size']
    config['dropout_mlp'] = loaded_config['dropout_mlp']
    config['dropout_mpn'] = loaded_config['dropout_mpn']
    if 'n_mt_blocks' in loaded_config:
        config['n_mt_blocks'] = loaded_config['n_mt_blocks']

    if loaded_config.get('use_rxn_class', False):
        config['n_atom_feat'] = ATOM_FDIM + len(RXN_CLASSES)
    else:
        config['n_atom_feat'] = ATOM_FDIM
    config['n_bond_feat'] = BOND_FDIM

    toggles = {}
    toggles['use_attn'] = loaded_config.get('use_attn', False)
    toggles['use_prev_pred'] = loaded_config.get('use_prev_pred', True)
    toggles['use_rxn_class'] = loaded_config.get('use_rxn_class', False)

    if 'n_heads' in loaded_config:
        config['n_heads'] = loaded_config['n_heads']
    config['embed_bias'] = False

    if loaded_config.get('use_h_labels', True):
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "h_labels")
    else:
        lg_dict = joblib.load(os.path.join(loaded_config['data_dir'], "train", "without_h_labels", loaded_config['vocab_file']))
        lg_tensor_dir = os.path.join(loaded_config['data_dir'], "train", "without_h_labels")

    if loaded_config.get('use_rxn_class', False):
        lg_tensor_dir = os.path.join(lg_tensor_dir, "with_rxn")
    else:
        lg_tensor_dir = os.path.join(lg_tensor_dir, "without_rxn")

    lg_vocab = Vocab(lg_dict)

    model_config['config'] = config
    model_config['toggles'] = toggles
    model_config['lg_vocab'] = lg_vocab
    return model_config

CONFIG_FNS = {
    'single_edit': build_edits_config,
    'multi_edit': build_edits_config,
    'single_shared': build_shared_edit_config,
    'lg_classifier': build_lg_classifier_config,
    'lg_ind': build_lg_ind_config
}

def build_model(loaded_config, device='cpu'):
    config_fn = CONFIG_FNS.get(loaded_config['model'])
    model_config = config_fn(loaded_config)

    if loaded_config['mpnn'] == 'graph_feat':
        encoder_name = 'GraphFeatEncoder'
    elif loaded_config['mpnn'] == 'wln':
        encoder_name = 'WLNEncoder'
    elif loaded_config['mpnn'] == 'gtrans':
        encoder_name = 'GTransEncoder'

    model_class = MODEL_ATTRS.get(loaded_config['model'])[0]
    model = model_class(**model_config, encoder_name=encoder_name, device=device)
    return model
