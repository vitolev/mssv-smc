#!/bin/bash
#SBATCH --job-name=pmmh_bpf_synth_T_200
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=pmmh_bpf_synth_T_200.out
#SBATCH --time=04:00:00

echo "Job started on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Move to project directory
cd ~/mssv-smc   # Adjust this path to your project directory

# Set project root
export ROOT_DIR=$(pwd)
# Make src visible to Python
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# Load Python module
module load Python/3.13.1-GCCcore-14.2.0    # Adjust the module name and version if necessary
echo "Loaded Python module."

# Activate virtual environment
source venv/bin/activate    # Make sure virtual environment is set up and adjust path if necessary
echo "Activated virtual environment. Starting Python script."

# Run script
python scripts/pmmh_bpf_synth_T_200.py \
    --N 3000 \
    --K 2 \
    --M 10000 \
    --C 8 \
    --burnin 5000