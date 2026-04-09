#!/bin/bash
#SBATCH --job-name=mssv_pmmh_bpf_synth_%j
#SBATCH --partition=all
#SBATCH --account=fri-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=mssv_pmmh_bpf_synth_%j.out
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
python experiments/mssv_pmmh_bpf_synth/mssv_pmmh_bpf_synth.py