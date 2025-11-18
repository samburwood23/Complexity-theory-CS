# main.tf
# This would define providers, regions, and call modules

resource "aws_vpc" "region_a_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = { Name = "app-vpc-region-a" }
}

resource "aws_vpc" "region_b_vpc" {
  cidr_block = "10.1.0.0/16"
  tags = { Name = "app-vpc-region-b" }
}

module "app_tier_region_a" {
  source = "./modules/app_tier"
  vpc_id = aws_vpc.region_a_vpc.id
  region = "us-east-1"
}

module "app_tier_region_b" {
  source = "./modules/app_tier"
  vpc_id = aws_vpc.region_b_vpc.id
  region = "us-west-2"
}
