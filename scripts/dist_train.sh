#!/usr/bin/env bash

set -x
NGPUS=$1
PORT=$2

python -m torch.distributed.launch --master_port=${PORT} --nproc_per_node=${NGPUS} main.py --sync_bn

