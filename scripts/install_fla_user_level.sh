#!/usr/bin/env bash
# Install flash-linear-attention (fla) into a plain directory, using the CONTAINER's own pip, without root and without rebuilding the image.
# The directory is later put on the container's PYTHONPATH (see examples/sbatch_snippet.sh).
#
# Usage:
#   SIF=/path/to/vllm-openai.sif TARGET=/path/on/shared/fs/pylib bash scripts/install_fla_user_level.sh
# Optional: FLA_VERSION (default 0.5.2), RUNTIME (singularity | apptainer; default: whichever is on PATH)
set -euo pipefail
SIF=${SIF:?set SIF to the container image}
TARGET=${TARGET:?set TARGET to an install directory on a shared filesystem}
FLA_VERSION=${FLA_VERSION:-0.5.2}
RUNTIME=${RUNTIME:-$(command -v apptainer || command -v singularity)}
[ -n "$RUNTIME" ] || { echo "neither apptainer nor singularity found"; exit 1; }

mkdir -p "$TARGET"
echo "[install] runtime=$RUNTIME sif=$SIF target=$TARGET fla==$FLA_VERSION"
# --no-deps: torch, triton, einops and transformers already come with the container; letting pip resolve them could put a second
# torch/triton into TARGET, which would then shadow the container's own builds.
"$RUNTIME" exec "$SIF" pip install --no-deps --target "$TARGET" "flash-linear-attention==$FLA_VERSION" "fla-core==$FLA_VERSION"

# import check (works without a GPU; fla may print a CPU-device warning on a login node)
PYTHONPATH="$TARGET" "$RUNTIME" exec "$SIF" python3 -c "import fla, torch, triton; print('fla', fla.__version__, '| torch', torch.__version__, '| triton', triton.__version__)"
du -sh "$TARGET"
