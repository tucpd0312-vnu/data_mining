import kagglehub

# Download latest version
path = kagglehub.dataset_download("data/hourly-energy-consumption")

print("Path to dataset files:", path)