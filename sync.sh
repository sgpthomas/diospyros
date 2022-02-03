#!/usr/bin/env sh

set -x

ADDR=$1

# some other commands
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=target \
      --exclude=*-results \
      -e "ssh -i ~/.ssh/thelio.pem" \
      `pwd`/ "ubuntu@$ADDR:~/$dir/"
