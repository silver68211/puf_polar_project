
"""
AFF3CT polar SRAM-PUF Monte Carlo simulation.

Supported decoders:
    SC
    DSCF
    SCL

All simulation settings are loaded from config.py.

Normalized decoding complexity:
    SC   = 1
    DSCF = total SC passes / decoded frames
    SCL  = L (nominal SC-equivalent complexity)

Each decoder has a separate CSV file containing:
    p, BLER, normalized_complexity

All progress reporting is performed by the C++ engine.
No plots or Python progress messages are generated.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from config import SimConfig
from Enrolment import SRAMPUFEnrollment
from keycheck import KeyChecker
from utils import generate_5g_ranking, polar_transform_bits
from puf_aff3ct import BSCMonteCarloPolar


CSV_COLUMNS = (
    "p",
    "BLER",
    "normalized_complexity",
)


# ============================================================
# 1. Select decoder configurations
# ============================================================


def get_decoder_runs(cfg):
    """Return the selected decoder and list-size combinations."""

    selected = cfg.decoder

    # Also accept a single decoder string for compatibility.
    if isinstance(selected, str):
        selected = (selected,)

    selected = tuple(
        str(name).strip().upper()
        for name in selected
    )

    valid_decoders = {"SC", "DSCF", "SCL"}

    if not selected:
        raise ValueError(
            "At least one decoder must be selected."
        )

    if any(name not in valid_decoders for name in selected):
        raise ValueError(
            "Valid decoders are SC, DSCF, and SCL."
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "Duplicate decoder selections are not allowed."
        )

    runs = []

    for name in selected:

        if name in ("SC", "DSCF"):
            runs.append((name, 1))

        elif name == "SCL":

            list_sizes = tuple(
                int(L) for L in cfg.list_sizes
            )

            if not list_sizes:
                raise ValueError(
                    "SCL list_sizes cannot be empty."
                )

            if len(set(list_sizes)) != len(list_sizes):
                raise ValueError(
                    "Duplicate SCL list sizes are not allowed."
                )

            for L in list_sizes:

                if L < 1 or (L & (L - 1)):
                    raise ValueError(
                        "SCL list sizes must be positive powers of two."
                    )

                runs.append(("SCL", L))

    return runs


# ============================================================
# 2. Construct the polar code
# ============================================================

def construct_polar_code(cfg, n, k):

    construction = str(
        getattr(cfg, "construction", "BSC")
    ).strip().upper()

    if construction == "5G":

        frozen, info = generate_5g_ranking(k, n)

    elif construction in ("BSC", "BHATTACHARYYA"):

        from constructions import bhattacharyya_design

        design_p = float(cfg.design_p)

        if not (0.0 < design_p < 0.5):
            raise ValueError(
                "design_p must be in (0, 0.5)."
            )

        frozen, info, _ = bhattacharyya_design(
            n,
            k,
            channel="BSC",
            design_p=design_p,
        )

        construction = "BSC"

    else:
        raise ValueError(
            "construction must be 5G or BSC."
        )

    frozen = np.sort(
        np.asarray(frozen, dtype=np.int32)
    )

    info = np.sort(
        np.asarray(info, dtype=np.int32)
    )

    if (
        len(frozen) != n - k
        or len(info) != k
        or not np.array_equal(
            np.sort(np.concatenate((frozen, info))),
            np.arange(n)
        )
    ):
        raise ValueError(
            "Invalid frozen/information-bit partition."
        )

    return frozen, info, construction


# ============================================================
# 3. Prepare output CSV
# ============================================================

def create_csv(out_dir, filename):

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = out_dir / filename

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()

    return path


def append_csv(path, row):

    with path.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow({
            "p": row["p"],
            "BLER": row["BLER"],
            "normalized_complexity":
                row["normalized_complexity"],
        })


# ============================================================
# 4. Verify Monte Carlo results
# ============================================================

def verify_result(result, decoder, L, trials, nlogn, max_blocks):

    blocks = int(result["blocks"])
    units = int(result["total_sc_equivalent_units"])
    passes = result["total_sc_passes"]

    complexity = float(
        result["normalized_complexity"]
    )

    if not (0 < blocks <= max_blocks):
        raise RuntimeError(
            "Invalid Monte Carlo block count."
        )

    if result["nlogn_per_pass"] != nlogn:
        raise RuntimeError(
            "Incorrect N log2(N) operation count."
        )

    if (
        result["total_sc_equivalent_ops"]
        != units * nlogn
    ):
        raise RuntimeError(
            "Inconsistent SC-equivalent operations."
        )

    if not np.isclose(
        complexity,
        units / blocks,
        rtol=1e-12,
        atol=1e-12
    ):
        raise RuntimeError(
            "Incorrect normalized complexity."
        )

    if decoder == "SC":

        if passes != blocks or units != blocks:
            raise RuntimeError(
                "SC must execute one pass per frame."
            )

        if complexity != 1.0:
            raise RuntimeError(
                "SC normalized complexity must be 1."
            )

    elif decoder == "DSCF":

        if passes != units:
            raise RuntimeError(
                "DSCF complexity counter mismatch."
            )

        if not (
            blocks <= units <= blocks * (1 + trials)
        ):
            raise RuntimeError(
                "Invalid DSCF decoding attempts."
            )

    elif decoder == "SCL":

        if units != blocks * L:
            raise RuntimeError(
                "Incorrect nominal SCL complexity."
            )

        if complexity != float(L):
            raise RuntimeError(
                "SCL normalized complexity must equal L."
            )

        if result["complexity_model"] != "nominal_L":
            raise RuntimeError(
                "Incorrect SCL complexity model."
            )


# ============================================================
# 5. Main simulation
# ============================================================

def simulate(cfg: SimConfig) -> list[dict]:

    # ------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------

    n = 1 << int(cfg.n_log2)
    k = int(cfg.key_length)

    q = float(cfg.puf_one_probability)

    p_values = tuple(
        float(p) for p in cfg.p_values
    )

    trials = int(cfg.num_trials)
    alpha = float(cfg.alpha)

    min_errors = int(cfg.min_block_errors)
    max_blocks = int(cfg.max_blocks)

    checker_kind = str(cfg.checker).strip().lower()

    seed = int(
        getattr(cfg, "seed", 2026)
    )

    scramble = bool(
        getattr(cfg, "scramble_codeword", False)
    )

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else 1
    )

    threads = int(
        getattr(cfg, "num_threads", available_cpus)
    )

    out_dir = Path(cfg.out_dir)

    # ------------------------------------------------------
    # Validate parameters
    # ------------------------------------------------------

    if not (1 <= k < n):
        raise ValueError("Require 0 < K < N.")

    if not (0.0 < q < 1.0):
        raise ValueError(
            "puf_one_probability must be in (0, 1)."
        )

    if cfg.all_zero_device and q != 0.5:
        raise ValueError(
            "All-zero baseline requires q=0.5."
        )

    if not p_values or any(
        not (0.0 <= p < 0.5) for p in p_values
    ):
        raise ValueError(
            "BSC probabilities must be in [0, 0.5)."
        )

    if threads < 1:
        raise ValueError(
            "num_threads must be positive."
        )

    if trials < 0 or alpha <= 0:
        raise ValueError(
            "Invalid DSCF parameters."
        )

    if min_errors < 1 or max_blocks < 1:
        raise ValueError(
            "Invalid Monte Carlo stopping limits."
        )

    if getattr(cfg, "compare_code_offset", False):
        raise NotImplementedError(
            "compare_code_offset is not supported "
            "by this native Monte Carlo engine."
        )

    if checker_kind not in ("hash", "crc"):
        raise ValueError(
            "checker must be hash or crc."
        )

    if checker_kind == "hash":

        if str(cfg.hash_name).lower() != "sha256":
            raise ValueError(
                "Native checker supports SHA-256 only."
            )

        if not (
            8 <= int(cfg.hash_bits) <= 256
            and int(cfg.hash_bits) % 8 == 0
        ):
            raise ValueError(
                "hash_bits must be byte-aligned "
                "and between 8 and 256."
            )

    if checker_kind == "crc":

        if str(cfg.crc_degree).upper() != "CRC16":
            raise ValueError(
                "Native checker supports CRC16 only."
            )

    decoder_runs = get_decoder_runs(cfg)

    # ------------------------------------------------------
    # Polar-code construction
    # ------------------------------------------------------

    frozen, info, construction = construct_polar_code(
        cfg, n, k
    )

    # ------------------------------------------------------
    # Enroll one fixed SRAM-PUF device
    # ------------------------------------------------------

    rng = np.random.default_rng(seed)

    checker = KeyChecker(
        method=checker_kind,
        hash_name=cfg.hash_name,
        hash_bits=int(cfg.hash_bits),
        crc_degree=cfg.crc_degree,
    )

    enrollment = SRAMPUFEnrollment(
        n,
        frozen,
        checker=checker,
        biased=(q != 0.5),
        prob_one=q,
    )

    if cfg.all_zero_device:

        device = enrollment.enroll(
            x=np.zeros((1, n), dtype=np.uint8),
            rng=rng,
        )

    else:

        device = enrollment.enroll(
            batch_size=1,
            rng=rng,
        )

    # ------------------------------------------------------
    # Prepare physical response, coset and secret
    # ------------------------------------------------------

    x = device.x[0].astype(np.uint8).tolist()

    z = polar_transform_bits(
        device.frozen_full
    )[0].astype(np.uint8).tolist()

    secret = device.key[0].astype(np.uint8).tolist()

    public_tag = device.tag[0].astype(np.uint8).tolist()

    checker_id = (
        "sha256"
        if checker_kind == "hash"
        else "crc16"
    )

    # ------------------------------------------------------
    # Output directory and filename prefix
    # ------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prefix = (
        f"BSC_N{n}_K{k}_{construction}_"
        f"{checker_kind}"
    )

    if checker_kind == "hash":
        prefix += f"{int(cfg.hash_bits)}"

    all_rows = []

    # ------------------------------------------------------
    # Run each decoder independently
    # ------------------------------------------------------
    print("=" * 120)
    print("AFF3CT POLAR SRAM-PUF MONTE CARLO")
    print("=" * 120)

    print(f"N={n}, K={k}")
    print(f"Construction: {construction}")
    print(f"Decoder: {decoder_runs}")
    print(f"DSCF: T={trials}, alpha={alpha}")
    print(f"Checker: {checker_kind}")
    print(f"Threads: {threads}")
    print(f"Scrambling: {scramble}")
    print(f"PUF P(X=1): {q}")
    print(f"Seed: {seed}")

    print(
        "Normalized complexity = "
        "total SC passes / decoded frames"
    )

    
    for decoder_name, L in decoder_runs:
        print("=" * 120)
        # ------------------------------------------
        # Decoder-specific CSV filename
        # ------------------------------------------

        if decoder_name == "SC":

            filename = f"{prefix}_SC.csv"
            label = "SC"

        elif decoder_name == "DSCF":

            filename = (
                f"{prefix}_DSCF_T{trials}.csv"
            )

            label = "DSCF"

        else:

            filename = (
                f"{prefix}_SCL_L{L}.csv"
            )

            label = f"SCL_L{L}"

        csv_path = create_csv(
            out_dir,
            filename
        )

        # ------------------------------------------
        # Initialize the native C++ engine
        # ------------------------------------------

        engine = BSCMonteCarloPolar(
            n=n,
            frozen_positions=frozen.tolist(),
            x=x,
            z=z,
            secret=secret,
            expected_tag=public_tag,
            decoder=decoder_name,
            checker=checker_id,
            hash_bits=int(cfg.hash_bits),
            prob_one=q,
            num_threads=threads,
            alpha=alpha,
            num_trials=trials,
            scramble_codeword=scramble,
            seed=seed,
            list_size=L,
        )

        # ------------------------------------------
        # Iterate over BSC probabilities
        # ------------------------------------------

        for p in p_values:

            # C++ handles all live terminal output.
            result = engine.run(
                p=p,
                min_errors=min_errors,
                max_blocks=max_blocks,
            )

            # --------------------------------------
            # Validate complexity calculations
            # --------------------------------------

            verify_result(
                result=result,
                decoder=decoder_name,
                L=L,
                trials=trials,
                nlogn=n * int(cfg.n_log2),
                max_blocks=max_blocks,
            )

            # --------------------------------------
            # Store only requested CSV columns
            # --------------------------------------

            row = {
                "p": p,
                "BLER": float(result["BLER"]),
                "normalized_complexity": float(
                    result["normalized_complexity"]
                ),
            }

            # Save immediately after each p.
            append_csv(csv_path, row)

            # Keep full in-memory identification
            # for possible later processing.
            all_rows.append({
                "decoder": label,
                **row,
            })

    return all_rows


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    simulate(SimConfig())
