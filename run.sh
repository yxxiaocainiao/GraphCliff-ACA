#!/bin/bash
# ========================================
# Multi-GPU Training Launcher with WandB Safety
#  - One process per GPU at a time
#  - Staggered start to avoid WandB conflicts
# ========================================

NUM_GPUS=8

DATASETS=(
     "CHEMBL237_Ki" "CHEMBL4203_Ki" "CHEMBL2034_Ki" "CHEMBL233_Ki" "CHEMBL4616_EC50"
     "CHEMBL287_Ki" "CHEMBL218_EC50" "CHEMBL264_Ki" "CHEMBL219_Ki"
     "CHEMBL2835_Ki" "CHEMBL2147_Ki" "CHEMBL231_Ki" "CHEMBL3979_EC50"
     "CHEMBL237_EC50" "CHEMBL244_Ki" "CHEMBL4792_Ki" "CHEMBL1871_Ki"
     "CHEMBL262_Ki" "CHEMBL2047_EC50" "CHEMBL239_EC50"
     "CHEMBL2971_Ki" "CHEMBL204_Ki" "CHEMBL214_Ki" "CHEMBL1862_Ki"
     "CHEMBL234_Ki" "CHEMBL238_Ki" "CHEMBL235_EC50" "CHEMBL4005_Ki"
     "CHEMBL236_Ki" "CHEMBL228_Ki"
)

TOTAL=${#DATASETS[@]}
echo "[INFO] Starting training on ${TOTAL} datasets (GPUs=${NUM_GPUS})"

# job counter
job_id=0

# Assign datasets to GPUs
for dataset in "${DATASETS[@]}"; do
    gpu=$((job_id % NUM_GPUS))
    
    # Create a unique run name
    echo "[INFO] Dataset=${dataset} → GPU${gpu}"

    # Run on the assigned GPU (background job)
    CUDA_VISIBLE_DEVICES=$gpu \
    python main.py \
        --project_name "GraphCliff" \
        --dataset "${dataset}" \
        --epochs 100 \
        -w &

    job_id=$((job_id + 1))
    
    # Add delay to avoid WandB init conflicts
    if (( job_id < TOTAL )); then
        echo "[INFO] Waiting 3s before launching the next job..."
        sleep 3
    fi

    # If all GPU slots are filled, wait for jobs to finish
    if (( job_id % NUM_GPUS == 0 )); then
        wait
        echo "---- [INFO] Previous ${NUM_GPUS} jobs completed ----"
    fi
done

# Wait for remaining jobs
wait
echo "========================================"
echo " ✅ Training completed on all datasets."
echo " Experiment group: experiment_${EXPERIMENT_ID}"
echo "========================================"
