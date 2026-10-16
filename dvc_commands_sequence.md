---
title: DVC Commands Used
subtitle: Short sequence for the tutorial project
---

# DVC Commands Used

- Track data
- Reproduce training
- Push artifacts

# 1. Initialize / Setup

- `dvc init`
- Creates `.dvc/`
- Enables DVC tracking in the Git repo
- In this repo, `.dvc/` already existed

# 2. Get And Track Data

- `dvc get https://github.com/treeverse/dataset-registry tutorials/versioning/data.zip`
- Downloads the tutorial dataset zip
- `dvc add data`
- Creates or updates `data.dvc`
- Stores data in the DVC cache

# 3. Reproduce Pipeline

- `dvc repro`
- Runs the `train` stage from `dvc.yaml`
- Creates model weights, `.npy` files, `metrics.csv`, and `dvc.lock`
- If unchanged: data and pipelines are up to date

# 4. Configure Remote And Push

- `dvc remote add -d localremote /tmp/dvc-example-remote`
- Configures a default local DVC remote
- `dvc push`
- Uploads cached data/model artifacts to the remote
- Latest result: everything is up to date

# 5. Useful Repeat Commands

- `source .env/bin/activate`
- `dvc status`
- `dvc repro`
- `dvc push`
