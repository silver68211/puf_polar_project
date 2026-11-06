# Polar codes for SRAM-PUF key regeneration

Based on Chen et al., GLOBECOM 2017, *A Robust SRAM-PUF Key Generation Scheme Based on Polar Codes*. Also includes a **biased-source baseline**, motivated by Chen & Willems, IEEE IoT Journal (2019). The full secrecy-aware biased-PUF polar construction from 2019 is **not** claimed to be reproduced.

## Files

- `utils.py`: all original utilities plus BSC/AWGN/BEC and a fast GF(2) polar transform.
- `Enrolment.py`: new SRAM enrollment, biased/unbiased generation, key/helper split and paired code-offset scheme.
- `keycheck.py`: SHA-256 / SHA3-256 / SHA-512 / BLAKE2s hash or CRC as a **separately stored key-confirmation tag**.
- `DSCFlip.py`: original SC recursive algorithm modified to accept per-word nonzero helper/frozen bits; original order-1 DSCF metric retained in log domain, candidate handling and batch ordering fixed; hash/CRC checking enabled.
- `constructions.py`: original constructions plus Bhattacharyya reliability ordering; exact for BEC, BSC/AWGN conservative upper-bound approximations.
- `crc.py`: retained original CRC module for compatibility (new external CRC tag is in `keycheck.py`).
- `sim.py`: syndrome simulation with decoder choice, channel choice, biased/unbiased source, optional paired code-offset comparison, CSV/PNG outputs.
- `test_puf.py`: unit tests and noiseless decoder regression tests.

## Requirements

Python compatible with your TensorFlow installation (typically Python 3.10/3.11/3.12 depending on TF version), NumPy, TensorFlow, SciPy, Matplotlib, Pandas, pytest. Installation in an isolated environment is recommended.

```bash
python -m pip install numpy scipy pandas matplotlib pytest tensorflow
python -m pytest -q test_puf.py
```

## Basic commands

```bash
# Fast functional smoke test: N=64, K=8, a few hundred samples
python sim.py --n-log2 6 --key-length 8 --points 0.14 0.15 --batch-size 16 --max-blocks 160 --min-errors 10

# The assignment's (1024,128), BSC p=0.14..0.30, SC, hash
python sim.py

# Hash-aided order-one DSCF: no extra check bits inserted into the key
python sim.py --decoder DSCF --trials 10 --points 0.15 0.20

# Alternate hash algorithm
python sim.py --decoder DSCF --hash-name sha3_256 --hash-bits 128 --points 0.15

# Nonuniform source P(X=1)=0.4: bias-aware LLR, NOT secrecy-certified
python sim.py --biased --prob-one 0.4 --points 0.15 0.20

# External CRC checker in place of hash (different from CRC-appended coding)
python sim.py --decoder DSCF --checker crc --crc-degree CRC16 --no-compare

# Alternative channels
python sim.py --channel BEC --points 0.10 0.20 0.30
python sim.py --channel AWGN --points 0.0 1.0 2.0
```

## What is different from communications?

Enrollment chooses the random SRAM state `X` (or accepts provided `X`), computes `C = X G_N`, and stores `helper = C[F]` while using `key = C[info]`. During reconstruction, `Y` is a noisy observation of the **same** `X`. The SC frozen leaves must now use the **per-enrollment helper values** rather than zero. The decoder reconstructs `key_hat = C_hat[info]`. The hash/CRC is computed **from the key** and stored for verification, not added to the polar information set; the key length remains `K`.

### Biased case

For q=P(X=1) and BSC crossover p, the correct observation posterior LLR before the polar transform is

`log(P(X=0|Y)/P(X=1|Y)) = log((1-q)/q) + (1-2Y)*log((1-p)/p)`.

For AWGN use `log((1-q)/q) + 2Y/sigma^2`. For BEC erased observations use `log((1-q)/q)`; visible bits use ±high LLR. With q=0.5 the prior term vanishes.

**IMPORTANT:** the bias-aware channel LLR is a reconstruction improvement, **not** a secrecy result. In Chen & Willems (2019) a specifically designed polar construction controls both leakage and reliability; merely using the 2017 syndrome splitting with biased data can leak information about the key. A public hash also reveals a key-derived verification value, and must be included in threat-model/security analysis. Do not claim zero leakage for `(W,Hash(S))`.

## Validation status

- Source inspected and all files syntax-checked in the generation environment.
- NumPy-only transform/channel/enrollment/checker tests were run independently.
- The provided execution environment does **not** contain TensorFlow, so the TensorFlow SC/DSCF runtime tests and Fig. 4 numerical reproduction **must be run in your TensorFlow environment** before interpreting performance. No unverified failure-rate claims are made.

## Reproduction caveats

- The Bhattacharyya BSC formula used here is a conservative upper-bound construction, **not** a reproduction of the exact 2017 paper's undocumented reliability set. The resulting SC curve may differ in detail from its Fig. 4.
- `omega>1` was not implemented in the original DSCF and is explicitly rejected instead of silently producing order-1 results.
- The code-offset equivalence is expected in **error-rate distribution** under uniform symmetric-channel assumptions. For BEC/LLR ties, per-trial SC decisions can differ even when the overall error rates agree. In biased mode the simplified code-offset comparison is disabled automatically.
- Pure Monte Carlo cannot practically establish `10^-9` with usual budgets. Zero observed failures is only an upper bound, not a measured zero error probability.
