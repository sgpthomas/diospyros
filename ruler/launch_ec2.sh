

# r5n.xlarge

aws ec2 run-instances \
    --image-id ami-08d14340baa856967 \
    --security-group-ids sg-0da7694c867e84ae3 \
    --count 1 \
    --instance-type r5n.xlarge \
    --key-name thelio

aws ec2 describe-instances \
  --query "Reservations[*].Instances[*].PublicIpAddress" \
  --output=text

aws ec2 describe-instances \
  --query "Reservations[*].Instances[*].PublicDnsName" \
  --output=text

aws ec2 describe-images --owner self
