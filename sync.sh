#!/usr/bin/env sh

set -x

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
      --exclude=time-ablation* \
      --exclude=variable_dup_abl_n_ops \
      --exclude=vd_abl_n_op2_vec_size \
      --exclude=vec \
      --exclude=t*.json \
      -e "ssh -i ~/.ssh/thelio.pem" \
      `pwd`/ "ubuntu@$ADDR:~/$dir/"
