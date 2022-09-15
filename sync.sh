#!/usr/bin/env sh

set -x

SCRIPT=$(readlink -f "$0")
PROJ_DIR=$(dirname "$SCRIPT")
ADDR=$1

# some other commands
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=ruler/target \
      --exclude=src/dios-egraphs/target \
      --exclude=*-results \
      --exclude=experiments/feb* \
      --exclude=experiments/res* \
      --exclude=experiments/*.csv \
      --exclude=experiments/*.org \
      --exclude=experiments/*.json \
      --exclude=experiments/*.png \
      --exclude=time-ablation* \
      --exclude=variable_dup_abl_n_ops \
      --exclude=vd_abl_n_op2_vec_size \
      --exclude=web-demo \
      -e "ssh -i ~/.ssh/thelio.pem" \
      $PROJ_DIR/ "ubuntu@$ADDR:~/$dir/"
