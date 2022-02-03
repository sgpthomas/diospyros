#!/usr/bin/env sh

set -x

ADDR=$1

# some other commands
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=target \
      --exclude=*-results \
      --exclude=experiments/feb* \
      --exclude=experiments/res* \
      --exclude=experiments/*.csv \
      --exclude=experiments/*.org \
      -e "ssh -i ~/.ssh/thelio.pem" \
      `pwd`/ "ubuntu@$ADDR:~/$dir/"
