#!/bin/bash
#SBATCH --job-name=fla_example
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
# Example: run a Hugging Face script inside the container, with the user-level fla kernels switched on by FLA=1.
# Submit:  sbatch --export=ALL,FLA=1,SIF=/path/to/vllm-openai.sif,PYLIB=/path/to/pylib examples/sbatch_snippet.sh <your_script.py> [args]
# Without FLA=1 the container's own environment is used unchanged (reference PyTorch path), so older runs stay reproducible.
set -euo pipefail
SIF=${SIF:?container image}
PYLIB=${PYLIB:?directory created by scripts/install_fla_user_level.sh}
RUNTIME=$(command -v apptainer || command -v singularity)

if [ "${FLA:-0}" = "1" ]; then
    # SINGULARITYENV_<NAME> / APPTAINERENV_<NAME> sets <NAME> inside the container
    export SINGULARITYENV_PYTHONPATH="$PYLIB" APPTAINERENV_PYTHONPATH="$PYLIB"
    echo "[env] fla kernels ON: $PYLIB"
else
    echo "[env] fla kernels OFF (reference implementation)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
# -B binds the shared filesystem that holds the model, PYLIB and the script into the container; adjust to your site
"$RUNTIME" exec --nv -B /scratch "$SIF" python3 "$@"
