set -x

ADDR=3.129.195.122

# some other commands
dir=diospyros
rsync -rP --exclude=.git \
      --exclude=target \
      -e "ssh -i ~/.ssh/thelio.pem" \
      ~/Research/$dir/ "ubuntu@$ADDR:~/$dir/"
