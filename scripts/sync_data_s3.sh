mkdir -p data

time aws s3 sync s3://wiris-ml-datasets/files data
