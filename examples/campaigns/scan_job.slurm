#!/bin/bash
#SBATCH --job-name=physai_scan
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
physai run scan examples/scan.yaml
