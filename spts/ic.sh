# spts/ic.sh

#!/bin/bash

# source /ljfs/local/basesoft/conda/etc/profile.d/conda.sh
# conda activate py369ljnp

set -eux

cd "C:/Users/admin/Desktop/SJW"

CONFIG=ictrain.yaml
# CONFIG=default.yaml


echo "===== TRAIN START $(date) ====="
echo "Config: $CONFIG"

MODEL=$(python -m app.train --config $CONFIG)
# sleep 60


echo "===== EVAL START $(date) ====="
echo "Config: $CONFIG, Model: $MODEL"

COMBO=$(python -m app.eval --config $CONFIG --model $MODEL)
# sleep 60


echo "===== BACKTEST START $(date) ====="
echo "Combo: $COMBO"

# python -m app.test --combo $COMBO
# sleep 60


echo "===== SH END $(date) ====="


# end of spts/ic.sh