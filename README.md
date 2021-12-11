# Learning Graph Models for Template-Free Retrosynthesis

## Setup

This assumes conda is installed on your system \
If conda is not installed, download the [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#).
If conda is installed, run the following commands:

```
echo 'export SEQ_GRAPH_RETRO=/path/to/dir/' >> ~/.bashrc
source ~/.bashrc
mkdir -p $SEQ_GRAPH_RETRO/datasets/

conda env create -f environment.yml
source activate seq_gr
python setup.py develop(or install)
```

## Datasets
The original and canonicalized files are provided under `datasets/uspto-50k/`

## Input Preparation

Before preparing inputs, we canonicalize the products. This can be done by running,

```
python data_process/canonicalize_prod.py --filename train.csv
python data_process/canonicalize_prod.py --filename eval.csv
python data_process/canonicalize_prod.py --filename test.csv
```
This step can also be skipped if the canonicalized files are already present.
The preprocessing steps now directly work with the canonicalized files.

#### 1. Reaction Info preparation
```
python data_process/parse_info.py --mode train
python data_process/parse_info.py --mode eval
python data_process/parse_info.py --mode test
```

#### 2. Prepare batches for Edit Prediction
```
python data_process/core_edits/bond_edits.py
```

#### 3. Prepare batches for Synthon Completion
```
python data_process/lg_edits/lg_classifier.py
python data_process/lg_edits/lg_tensors.py
```

## Run a Model
Trained models are stored in `experiments/`. You can override this by adjusting `--exp_dir` before training.
Model configurations are stored in `config/MODEL_NAME` 
where `MODEL_NAME` is one of `{single_edit, lg_classifier}`.

To run a model, 
```
python scripts/benchmarks/run_model.py --config_file config/MODEL_NAME/defaults.yaml
```
NOTE: We recently updated the code to use wandb for experiment tracking. You would need to setup [wandb](https://docs.wandb.ai/quickstart) before being able to train a model.

## Evaluate using a Trained Model

To evaluate the trained model, run
```
python scripts/eval/single_edit_lg.py --edits_exp EDITS_EXP --edits_step EDITS_STEP \
                                      --lg_exp LG_EXP --lg_step LG_STEP
```
This will setup a model with the edit prediction module loaded from experiment `EDITS_EXP` and checkpoint `EDITS_STEP` \
and the synthon completion module loaded from experiment `LG_EXP` and checkpoint `LG_STEP`.

## Reproducing our results
To reproduce our results, please run the command,
```
./eval.sh
```
This will display the results for reaction class unknown and known setting.
