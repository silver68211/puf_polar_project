#!/bin/bash
#SBATCH --job-name=puf-polar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=23:59:00
#SBATCH --partition=cpu-share
#SBATCH --qos=ece-normal

#SBATCH --chdir=/home/hnksm/puf_polar_project
#SBATCH --output=/home/hnksm/puf_polar_project/jobs/logs/puf_%j.out
#SBATCH --error=/home/hnksm/puf_polar_project/jobs/logs/puf_%j.err

set -euo pipefail

# ============================================================
# 1. Project and experiment selection
# ============================================================

# Supported decoders:
#   SC, DSCF, SCL_L1, SCL_L2, SCL_L4, SCL_L8, ...
#
# Supported enrollment schemes:
#   syndrome, code_offset, both
#
# Defaults to syndrome when PUF_SCHEME is omitted.

PROJECT_DIR="/home/hnksm/puf_polar_project"
LOG_DIR="${PROJECT_DIR}/jobs/logs"

CASE="${PUF_CASE:-}"
SCHEME="${PUF_SCHEME:-syndrome}"

# ------------------------------------------------------------
# Validate decoder selection
# ------------------------------------------------------------

if [[ -z "${CASE}" ]]; then
    echo "ERROR: PUF_CASE is required." >&2
    exit 1
fi

if [[ "${CASE}" != "SC" &&
      "${CASE}" != "DSCF" &&
      ! "${CASE}" =~ ^SCL_L[0-9]+$ ]]; then

    echo "ERROR: Invalid decoder case: ${CASE}" >&2
    exit 1
fi

# ------------------------------------------------------------
# Validate enrollment scheme
# ------------------------------------------------------------

case "${SCHEME}" in

    syndrome|transform)
        SCHEME="syndrome"
        ;;

    code_offset|both)
        ;;

    *)
        echo "ERROR: Invalid scheme: ${SCHEME}" >&2
        exit 1
        ;;
esac

cd "${PROJECT_DIR}"

# The logs directory must already exist when sbatch
# opens its output files.

mkdir -p "${LOG_DIR}"

# Unique results directory for each Slurm job.

RESULT_DIR="${PROJECT_DIR}/results/ece/job_${SLURM_JOB_ID}_${CASE}_${SCHEME}"

mkdir -p "${RESULT_DIR}"


# ============================================================
# 2. Activate Conda
# ============================================================

CONDA_SH=""

for candidate in \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh"
do

    if [[ -f "${candidate}" ]]; then
        CONDA_SH="${candidate}"
        break
    fi

done

if [[ -z "${CONDA_SH}" ]] &&
   command -v conda >/dev/null 2>&1; then

    CONDA_BASE="$(conda info --base)"
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialization not found." >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate puf_aff3ct


# ============================================================
# 3. CPU and numerical library settings
# ============================================================

# Native Monte Carlo uses C++ std::thread workers.
# Prevent numerical libraries from creating extra threads.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1


# ============================================================
# 4. Verify environment
# ============================================================

python -c "
from puf_aff3ct import BSCMonteCarloPolar
print('Native AFF3CT engine: OK')
"

echo
echo "============================================================"
echo "ECE HPC - POLAR SRAM-PUF MONTE CARLO"
echo "============================================================"

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Job name:           ${SLURM_JOB_NAME}"
echo "Decoder case:       ${CASE}"
echo "Enrollment scheme:  ${SCHEME}"

echo "Node:               $(hostname)"
echo "Start time:         $(date)"

echo "Partition:          ${SLURM_JOB_PARTITION}"
echo "Account:            ${SLURM_JOB_ACCOUNT}"

echo "Allocated CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Visible CPUs:       $(nproc)"
echo "CPU affinity:       $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"

echo "Python:             $(command -v python)"
echo "Conda environment:  ${CONDA_DEFAULT_ENV:-unknown}"

echo "Results:            ${RESULT_DIR}"

echo "============================================================"
echo


# ============================================================
# 5. Execute simulation
# ============================================================

# All scientific parameters are loaded from config.py.
#
# Per-job overrides:
#
#   enrollment_schemes
#   decoder
#   list_sizes
#   num_threads
#   out_dir
#
# config.py is never modified by this batch job.

set +e

srun \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    python -u - "${CASE}" "${SCHEME}" "${RESULT_DIR}" <<'PYTHON'

from __future__ import annotations

import json
import os
import socket
import sys

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from config import SimConfig
from sim_aff3ct import simulate


# ============================================================
# 6. Read job parameters
# ============================================================

case = sys.argv[1]
scheme = sys.argv[2]
result_dir = Path(sys.argv[3])


# ============================================================
# 7. Select decoder
# ============================================================

if case == "SC":

    selected_decoders = ("SC",)
    list_sizes = None

elif case == "DSCF":

    selected_decoders = ("DSCF",)
    list_sizes = None

elif case.startswith("SCL_L"):

    selected_decoders = ("SCL",)

    L = int(case.split("_L", 1)[1])

    if L < 1 or (L & (L - 1)):
        raise ValueError(
            "SCL list size must be a positive power of two."
        )

    list_sizes = (L,)

else:

    raise ValueError(
        f"Invalid decoder case: {case}"
    )


# ============================================================
# 8. Select enrollment schemes
# ============================================================

if scheme == "both":

    selected_schemes = (
        "syndrome",
        "code_offset",
    )

elif scheme in ("syndrome", "code_offset"):

    selected_schemes = (scheme,)

else:

    raise ValueError(
        f"Invalid enrollment scheme: {scheme}"
    )


# ============================================================
# 9. Determine available CPUs
# ============================================================

allocated_cpus = int(
    os.environ["SLURM_CPUS_PER_TASK"]
)

available_cpus = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else allocated_cpus
)

num_threads = min(
    allocated_cpus,
    available_cpus,
)

if num_threads < 1:
    raise RuntimeError("No CPUs available.")


# ============================================================
# 10. Override configuration
# ============================================================

# The original config.py is preserved.
# Changes are made only to the in-memory dataclass.

updates = {
    "decoder": selected_decoders,
    "enrollment_schemes": selected_schemes,
    "num_threads": num_threads,
    "out_dir": str(result_dir),
}

if list_sizes is not None:
    updates["list_sizes"] = list_sizes

cfg = replace(
    SimConfig(),
    **updates,
)

cfg.validate()


# ============================================================
# 11. Save complete configuration
# ============================================================

result_dir.mkdir(
    parents=True,
    exist_ok=True,
)

metadata = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "hostname": socket.gethostname(),
    "started_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "decoder_case": case,
    "enrollment_scheme_request": scheme,

    "allocated_cpus": allocated_cpus,
    "available_cpus": available_cpus,
    "simulation_threads": num_threads,

    "python_executable": sys.executable,

    "configuration": asdict(cfg),
}

settings_path = result_dir / "run_settings.json"

with settings_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 12. Print simulation parameters
# ============================================================

print("=" * 70)
print("Starting AFF3CT polar SRAM-PUF Monte Carlo")
print("=" * 70)

print(f"Decoder:      {selected_decoders}")
print(f"List sizes:   {list_sizes}")
print(f"Enrollment:   {selected_schemes}")

print(f"N:            {1 << cfg.n_log2}")
print(f"K:            {cfg.key_length}")

print(f"Construction: {cfg.construction}")
print(f"BSC p values: {cfg.p_values}")

print(f"Threads:      {num_threads}")
print(f"Seed:         {cfg.seed}")

print(f"Output:       {result_dir}")

print("=" * 70, flush=True)


# ============================================================
# 13. Run native AFF3CT Monte Carlo
# ============================================================

simulate(cfg)

PYTHON

EXIT_CODE=$?

set -e


# ============================================================
# 14. Final job information
# ============================================================

echo
echo "============================================================"
echo "ECE HPC SIMULATION FINISHED"
echo "============================================================"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Decoder:      ${CASE}"
echo "Enrollment:   ${SCHEME}"

echo "End time:     $(date)"
echo "Exit code:    ${EXIT_CODE}"

echo "Results:      ${RESULT_DIR}"

echo "============================================================"

exit "${EXIT_CODE}"
