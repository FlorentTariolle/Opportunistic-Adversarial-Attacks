#!/bin/bash
#SBATCH -J "ablation_naive"
#SBATCH -o slurm/logs/ablation_naive_%a.out
#SBATCH -e slurm/logs/ablation_naive_%a.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 16G
#SBATCH --time=08:00:00
#SBATCH --array=0-9

# Slurm job array: each task handles 10 images (100 images / 10 tasks).
# Submit:  sbatch slurm/ablation_naive.sl
# Monitor: tail -f slurm/logs/ablation_naive_*.out
# Resume:  sbatch again — CSV keys prevent duplicate work.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user -r requirements-hpc.txt
#   python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"

module purge
module load aidl/pytorch/2.6.0-cuda12.6

CHUNK=10
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

echo "Worker ${SLURM_ARRAY_TASK_ID}: images [${START}:${END}] starting at $(date)"
python -u benchmark_ablation_naive.py --image-start $START --image-end $END
echo "Worker ${SLURM_ARRAY_TASK_ID}: finished at $(date)"
