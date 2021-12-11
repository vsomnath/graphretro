#!/bin/bash

source activate seq_gr
EDITS_EXP="SingleEdit_10-02-2021--08-44-37"
EDITS_STEP="epoch_156"
LG_EXP="LGIndEmbed_18-02-2021--12-23-26"
LG_STEP="step_101951"

echo "Performance evaluation for reaction class unknown setting"
python scripts/eval/single_edit_lg.py \
    --edits_exp $EDITS_EXP \
    --lg_exp $LG_EXP \
    --edits_step $EDITS_STEP \
    --lg_step $LG_STEP \
    --exp_dir models

EDITS_EXP="SingleEdit_14-02-2021--19-26-20"
EDITS_STEP="step_144228"
LG_EXP="LGIndEmbedClassifier_18-04-2021--11-59-29"
LG_STEP="step_110701"

echo "Performance evaluation for reaction class known setting"
python scripts/eval/single_edit_lg.py \
    --edits_exp $EDITS_EXP \
    --lg_exp $LG_EXP \
    --edits_step $EDITS_STEP \
    --lg_step $LG_STEP \
    --exp_dir models/

