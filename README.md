
# Polar Codes for SRAM-PUF Key Regeneration

Monte Carlo simulation of SRAM-PUF key regeneration using **polar codes** and **SC, order-one DSCF, and SCL decoding**. The high-performance simulator is built on [AFF3CT](https://github.com/aff3ct/aff3ct) v4.7.1; a separate TensorFlow implementation provides SC/DSCF reference decoders.

The project compares **syndrome (transform)** and **code-offset** enrollment over the binary symmetric channel (BSC), including a biased-source baseline. It is inspired by [1] and [2], but does **not** reproduce the secrecy-aware construction of [2].

## 1. Requirements

- Linux, Git, a C++17 compiler, CMake ≥ 3.18, and Conda (or Miniforge).
- Python 3.10 and the Python dependencies below.
- AFF3CT **v4.7.1**, compiled as a C++ library.
- Optional: Slurm for HPC experiments.

## 2. Install and build

Run the following commands in a terminal. No administrator access is required.

**Get this repository and create the Python environment:**

```bash
git clone git@github.com:silver68211/puf_polar_project.git
cd puf_polar_project

conda create -n puf_aff3ct python=3.10 cmake pip -y
conda activate puf_aff3ct
export PYTHONNOUSERSITE=1
python -m pip install --no-user numpy scipy pandas matplotlib pytest pybind11 tensorflow
python -m pip check
```

TensorFlow and Matplotlib are included because the current Python utilities may import them even when the native simulator is used.

**Build and install AFF3CT v4.7.1:**

```bash
mkdir -p "$HOME/src" "$HOME/opt"
git clone --recursive --branch v4.7.1 \
  https://github.com/aff3ct/aff3ct.git "$HOME/src/aff3ct"

cmake -S "$HOME/src/aff3ct" -B "$HOME/src/aff3ct/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAFF3CT_COMPILE_EXE=OFF \
  -DAFF3CT_COMPILE_STATIC_LIB=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/aff3ct-v4.7.1"

cmake --build "$HOME/src/aff3ct/build" --parallel 4
cmake --install "$HOME/src/aff3ct/build"
```

**Build the native Python extension (from the project root):**

```bash
cmake -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/opt/aff3ct-v4.7.1" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DPUF_ENABLE_NATIVE_ARCH=OFF

cmake --build cpp/build --parallel 2
python -c "from puf_aff3ct import PolarSC, PolarDSCF, PolarSCLPUF, BSCMonteCarloPolar; print('Native bindings: OK')"
```

`PUF_ENABLE_NATIVE_ARCH=OFF` is recommended when building on one HPC node and running on another. If AFF3CT is installed elsewhere, replace `CMAKE_PREFIX_PATH` with its installation prefix.

## 3. Test the installation

From the project root, with `puf_aff3ct` activated:

```bash
export PYTHONNOUSERSITE=1
python -m pytest -q test_code_offset.py test_code_offset_native.py test_code_offset_noisy.py
```

These tests cover enrollment identities, noiseless native reconstruction, and bounded noisy-channel integration. They are **not** substitutes for statistically precise BLER measurements.

## 4. Configure a simulation

Edit `config.py` (`SimConfig`). For example, to compare both enrollment schemes on the **(1024, 128) 5G polar code**:

```python
n_log2 = 10                       # N = 1024
key_length = 128                  # K = 128
construction = "5G"              # Also: "BSC", "BHATTACHARYYA", "OURS"
enrollment_schemes = ("syndrome", "code_offset")
decoder = ("SC", "DSCF", "SCL")
list_sizes = (2, 4, 8)            # SCL only
num_trials = 10                   # DSCF extra attempts
p_values = (0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30)
puf_one_probability = 0.5         # 0.4 for biased-source experiments
checker = "hash"                  # "hash" or "crc"
hash_bits = 128
seed = 2026
```

These are configuration field values, **not a standalone Python script**. Set `min_block_errors`, `max_blocks`, `num_threads`, and `out_dir` in the same configuration. For quick tests, use small block limits; production simulations require enough observed errors for reliable BLER estimates. Custom `OURS` construction expects `codes/frozen_bits_N{N}_K{K}_our.csv`.

## 5. Run simulations

**Native AFF3CT (recommended for large experiments):**

```bash
python sim_aff3ct.py
```

The native driver runs the enrollment-scheme/decoder combinations selected in `config.py` and writes separate CSV files with `p`, `BLER`, and `normalized_complexity` to `out_dir`.

**Python/TensorFlow reference (SC and DSCF only):**

```bash
# First set decoder=("SC", "DSCF") in config.py
python sim_python.py
```

**Slurm / ECE HPC:** Ensure `jobs/aff3ct.sbatch` has a valid account, partition, QoS, project path, and Conda environment for your cluster. Create `jobs/logs` **before submission**:

```bash
mkdir -p jobs/logs

# Four jobs: both schemes, SC and DSCF
for scheme in syndrome code_offset; do
  for case in SC DSCF; do
    sbatch --job-name="${case}_${scheme}" \
      --export=ALL,PUF_CASE="$case",PUF_SCHEME="$scheme" \
      jobs/aff3ct.sbatch
  done
done

# Six jobs: both schemes, SCL list sizes 2, 4, and 8
for scheme in syndrome code_offset; do
  for L in 2 4 8; do
    sbatch --job-name="SCL_L${L}_${scheme}" \
      --export=ALL,PUF_CASE="SCL_L${L}",PUF_SCHEME="$scheme" \
      jobs/aff3ct.sbatch
  done
done

squeue -u "$USER"
```

Slurm runs use unique `results/ece/job_<ID>_<DECODER>_<SCHEME>/` directories, including CSV files and `run_settings.json`; logs are under `jobs/logs/`. `PUF_CASE=SCL` **is not valid**: use `SCL_L2`, `SCL_L4`, etc.

## 6. Enrollment and interpretation

Let `X` be the enrolled SRAM response, `Y = X ⊕ E` a noisy read, and `G_N` the polar transform.

| Scheme | Secret key | Public helper | Helper length |
| --- | --- | --- | --- |
| Syndrome / transform | `S = (X G_N)[info]` | `W = (X G_N)[frozen]` | `N − K` bits |
| Code offset | Independently sampled `S`; `C = encode(S)` | `Z = X ⊕ C` | `N` bits |

For syndrome enrollment, form the full offset `Z = encode(0_info, W_frozen)`. Both methods then decode `Y ⊕ Z` using a **zero-frozen** polar decoder. The hash/CRC verification tag is stored separately and is **not** counted among the `K` secret bits. The native DSCF and SCL decoders use that public tag for candidate selection; the true secret is used only to score reconstruction failures.

**Limitations:** For biased PUFs (`P(X=1) ≠ 0.5`), the current prior-adjusted LLR is a baseline, not a proof of optimal helper-conditioned decoding or key secrecy. Public helper data **and** key-confirmation tags can leak information. The native CRC checker must not be assumed bit-for-bit identical to the standalone polynomial CRC layer without verification. `normalized_complexity` is an SC-equivalent model (nominally `L` for SCL), not measured execution time. Zero observed failures does not imply zero failure probability.

## 7. Main files

| Path | Purpose |
| --- | --- |
| `config.py` | Simulation configuration and validation |
| `Enrolment.py`, `keycheck.py` | SRAM-PUF enrollment and public key verification |
| `constructions.py`, `utils.py` | Polar construction, transforms, channels, and utilities |
| `sim_aff3ct.py` | Native SC/DSCF/SCL Monte Carlo |
| `sim_python.py`, `DSCFlip.py`, `crc.py` | TensorFlow SC/DSCF reference implementation and CRC |
| `cpp/` | AFF3CT extensions, pybind11 bindings, and CMake build |
| `jobs/aff3ct.sbatch` | Slurm experiment launcher |
| `test_code_offset*.py` | Enrollment and native integration tests |

## References

[1] B. Chen, T. Ignatenko, F. M. J. Willems, R. Maes, E. van der Sluis, and G. N. Selimis, “A Robust SRAM-PUF Key Generation Scheme Based on Polar Codes,” in *Proc. IEEE Global Communications Conference (GLOBECOM)*, 2017, pp. 1–6. doi: [10.1109/GLOCOM.2017.8254007](https://doi.org/10.1109/GLOCOM.2017.8254007).

[2] B. Chen and F. M. J. Willems, “Secret Key Generation Over Biased Physical Unclonable Functions With Polar Codes,” *IEEE Internet of Things Journal*, vol. 6, no. 1, pp. 435–445, Feb. 2019. doi: [10.1109/JIOT.2018.2864594](https://doi.org/10.1109/JIOT.2018.2864594).

[3] A. Cassagne *et al.*, “AFF3CT: A Fast Forward Error Correction Toolbox!,” *SoftwareX*, vol. 10, Art. no. 100345, 2019. doi: [10.1016/j.softx.2019.100345](https://doi.org/10.1016/j.softx.2019.100345). See also the [official AFF3CT repository](https://github.com/aff3ct/aff3ct).
