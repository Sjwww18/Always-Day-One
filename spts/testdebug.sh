# spts/testdebug.sh

#!/bin/bash

# source /ljfs/local/basesoft/conda/etc/profile.d/conda.sh
# conda activate py369ljnp

set -eux

cd "C:/Users/admin/Desktop/SJW"

CONFIG=testdefault.yaml

echo "===== TRAIN START $(date) ====="
echo "Config: $CONFIG"

python -m app.test --config $CONFIG

echo "===== TRAIN END $(date) ====="


# end of spts/testdebug.sh