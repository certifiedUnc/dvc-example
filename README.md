# DVC Tutorial: Data and Model Versioning

This repo contains the code and DVC configuration for the DVC
[Data and Model Versioning tutorial](https://dvc.org/doc/tutorials/versioning).
It trains a cats-vs-dogs classifier, tracks the dataset with DVC, and captures
the generated model artifacts with a `dvc.yaml` pipeline.

## Setup

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Download the first dataset

```bash
dvc get https://github.com/treeverse/dataset-registry \
  tutorials/versioning/data.zip
unzip -q data.zip
rm -f data.zip
dvc add data
git add data.dvc .gitignore
```

This downloads 1000 training images and 800 validation images.

## Train with `dvc.yaml`

The training stage is configured in `dvc.yaml` and reads parameters from
`params.yaml`.

```bash
dvc repro
git add dvc.yaml dvc.lock params.yaml metrics.csv .gitignore
git commit -m "First model, trained with 1000 images"
git tag -a "v1.0" -m "model v1.0, 1000 images"
```

Tracked outputs:

- `model.weights.h5`
- `bottleneck_features_train.npy`
- `bottleneck_features_validation.npy`
- `metrics.csv`

## Add the second dataset version

```bash
dvc get https://github.com/treeverse/dataset-registry \
  tutorials/versioning/new-labels.zip
unzip -q new-labels.zip
rm -f new-labels.zip
dvc add data
dvc repro
git add data.dvc dvc.lock metrics.csv
git commit -m "Second model, trained with 2000 images"
git tag -a "v2.0" -m "model v2.0, 2000 images"
```

## Switch between versions

Restore the full first snapshot:

```bash
git checkout v1.0
dvc checkout
```

Restore only the first dataset while keeping the current code:

```bash
git checkout v1.0 data.dvc
dvc checkout data.dvc
```

Return to the latest branch afterward:

```bash
git checkout main
dvc checkout
```

If your default branch is named `master`, use `git checkout master` instead.

## Example: `dvc push`

`dvc push` uploads DVC-tracked data and model artifacts to a configured remote.
Use a local remote for a quick demo:

```bash
mkdir -p /tmp/dvc-example-remote
dvc remote add -d localremote /tmp/dvc-example-remote
git add .dvc/config
git commit -m "Configure local DVC remote"
dvc push
```

For a real project, configure cloud storage instead:

```bash
pip install "dvc[s3]"
dvc remote add -d storage s3://my-bucket/dvc-example
dvc push
git push --follow-tags
```

After another clone, recover the tracked files with:

```bash
dvc pull
```
