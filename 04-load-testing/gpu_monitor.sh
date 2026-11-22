#!/usr/bin/env bash
# Sample GPU util + memory every 1s into results/nvidia_smi.csv
mkdir -p results
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
  --format=csv -l 1 | tee results/nvidia_smi.csv
