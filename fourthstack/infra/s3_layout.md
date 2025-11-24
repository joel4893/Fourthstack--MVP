s3://fourthstack-data/
│
├── raw/
├── processed/
├── synthetic/
├── infused/
├── models/
├── reports/
└── temp/

Notes:
- `raw/` - ingested original datasets (immutable).
- `processed/` - cleaned and preprocessed datasets ready for training.
- `synthetic/` - generated synthetic datasets.
- `infused/` - datasets created by infusing synthetic into real data.
- `models/` - trained model artifacts, checkpoints.
- `reports/` - evaluation and monitoring reports.
- `temp/` - transient files, intermediate artifacts, safe to purge.

