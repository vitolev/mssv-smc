#!/bin/bash
#SBATCH --job-name=generate_data_1_regime
#SBATCH --partition=all
#SBATCH --account=fri-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=generate_data_1_regime.out
#SBATCH --time=00:10:00

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
python experiments/generate_data_1_regime/generate_data_1_regime.py \
    --T 2000 \
    --mu -1.0 \
    --phi 0.9 \
    --sigma_eta 0.1