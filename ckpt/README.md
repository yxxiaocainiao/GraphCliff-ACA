# Checkpoints

Pretrained GraphCliff checkpoints for the 30 MoleculeACE targets are hosted on
Zenodo and are **not** tracked in this repository.

**Record:** https://zenodo.org/records/21787497

```bash
# from the repository root
wget https://zenodo.org/records/21787497/files/graphcliff_ckpt.tar.gz
tar xzf graphcliff_ckpt.tar.gz
mv 250920_final_ckpt/* ckpt/
rmdir 250920_final_ckpt
```

Verify the archive before extracting:

```bash
md5sum graphcliff_ckpt.tar.gz
# 708921f6503864393107a332f4ac32fc  graphcliff_ckpt.tar.gz
```

The archive is 666.6 MB. It is released under CC BY 4.0, while the code in this
repository is MIT.

## Expected layout

```
ckpt/
├── CHEMBL204_Ki/
│   └── <TIMESTAMP>/
│       └── best_model.pt
├── CHEMBL214_Ki/
│   └── <TIMESTAMP>/
│       └── best_model.pt
└── ...                        30 targets in total
```

Each `best_model.pt` holds a single key, `model_state_dict`, with 6,021,198
float32 parameters matching the default architecture (`--hidden_size 256
--num_layers 3 --groups 4 --mid_K 3`).

## Usage

`main.py` globs `ckpt/<dataset>/*/best_model.pt`. When a checkpoint is present it
is loaded and evaluated on the test split instead of training from scratch:

```bash
python main.py --dataset CHEMBL204_Ki
```

The analysis scripts read the same location:

```bash
python -u analysis/topk_gate_alignment.py --mode all --k 1 3 5
python analysis/gate_heatmap.py
```

Training writes new checkpoints here as `ckpt/<dataset>/<timestamp>/best_model.pt`.
Everything in this directory except this README is git-ignored.
