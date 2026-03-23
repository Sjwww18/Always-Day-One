# spts/debug.sh

#!/bin/bash

# source /ljfs/local/basesoft/conda/etc/profile.d/conda.sh
# conda activate py369ljnp

set -eux

cd "C:/Users/admin/Desktop/SJW"

# CONFIG=ictrain.yaml
CONFIG=cnn.yaml


echo "===== TRAIN START $(date) ====="
echo "Config: $CONFIG"

python -m app.main --config $CONFIG

echo "===== TRAIN END $(date) ====="


# end of spts/debug.sh