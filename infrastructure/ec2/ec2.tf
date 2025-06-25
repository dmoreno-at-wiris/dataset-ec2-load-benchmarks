# resource "aws_subnet" "this" {
#   # vpc_id            = aws_vpc.example.id?
#   vpc_id            = "vpc-c16166bb"
#   cidr_block        = "172.31.16.0/20"
#   availability_zone = "us-east-1d"

# }

resource "aws_instance" "train_instance" {
  ami           = "ami-04824eb68e0de47b0"
  # ami           = "wiris-ml-g6-image"
  instance_type = "g6.2xlarge"
  key_name      = "wirisml-sharedkey"
  # subnet_id     = aws_subnet.this.id
  subnet_id     = "subnet-5943f814"
  # public_dns    = ?

  # user_data?     =
  # associate_public_ip_address?     =

  # NOTE:
  # Do not use volume_tags if you plan to manage block device tags outside the
  # aws_instance configuration, such as using tags in an aws_ebs_volume resource
  # attached via aws_volume_attachment. Doing so will result in resource cycling
  # and inconsistent behavior.
  # ---
  # volume_tags     =
  # OR
  # ebs_block_device?     =
  # OR
  # aws_volume_attachment? =

  # NOTE:
  # This option is only supported on creation of instance type that support CPU
  # Options CPU Cores and Threads Per CPU Core Per Instance Type - specifying
  # this option for unsupported instance types will return an error from the EC2
  # API.
  # ---
  # cpu_options {
  #   core_count       = 8
  # }
}

# Output the public DNS
output "instance_public_dns" {
  value = aws_instance.train_instance.public_dns
}
