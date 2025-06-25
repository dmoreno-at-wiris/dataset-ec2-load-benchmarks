terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.100"
    }
  }
  # s3 bucket to store the infrastructure state
  # https://www.terraform.io/docs/language/settings/backends/s3.html
  backend "s3" {}
}

# Default provider
provider "aws" {
  # Applies tags across all resources handled by this provider.
  # This is useful to replace redundant per-resource tags configurations.
  default_tags {
    tags = {
      team        = "ML"
      project     = var.project
      environment = var.environment
      repository  = "https://github.com/wiris/dataset-ec2-load-benchmarks"
    }
  }
}

# resources must be prefixed with the following variable
locals {
  prefix = var.project
}
