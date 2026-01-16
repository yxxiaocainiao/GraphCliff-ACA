# GraphCliff: Short-Long Range Gating for Subtle Differences but Critical Changes

This repository contains the reference code for **GraphCliff**.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running GraphCliff](#running-graphcliff)

# Environment Setup

1. **Install Miniconda or Anaconda** and ensure GPU drivers/CUDA match the versions listed in `environment.yml`

2. **Create the environment:**
   ```bash
   cd envs/
   conda env create -f graphcliff.yml
   conda activate graphcliff
   pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
   cd ..
   ```
  **Important:** Update the `prefix` at the bottom of the file so it points to your local conda installation before running the command.

# Running GraphCliff
**Running:** Set `NUM_GPUS` in run.sh to your number of GPUs. Load and validate the best checkpoint if it exists, else train on each data from scratch. 
   ```bash
   bash run.sh
   ```
