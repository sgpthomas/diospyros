set -x

ADDR=3.143.217.40

# some other commands
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=target \
      --exclude=*-results \
      -e "ssh -i ~/.ssh/thelio.pem" \
      ~/Research/$dir/ "ubuntu@$ADDR:~/$dir/"
