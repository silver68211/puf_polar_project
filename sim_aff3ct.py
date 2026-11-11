
"""
AFF3CT polar SRAM-PUF Monte Carlo simulation.

Enrollment schemes:
    syndrome    : PUF-derived key, N-K helper bits
    code_offset : Independent key, N helper bits

Supported decoders:
    SC
    DSCF
    SCL

Supported polar constructions:
    5G
    BSC
    OURS

Normalized decoding complexity:
    SC   = 1
    DSCF = average number of SC decoding passes
    SCL  = L (nominal SC-equivalent complexity)

Each enrollment scheme and decoder combination is
saved in a separate CSV file containing:

    p, BLER, normalized_complexity

Native C++ performs Monte Carlo simulation and
live progress reporting.

No figures are generated.
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
# 1. Select enrollment schemes
# ============================================================

def get_enrollment_schemes(cfg):

    selected = getattr(
        cfg,
        "enrollment_schemes",
        ("syndrome",),
    )

    if isinstance(selected, str):
        selected = (selected,)

    schemes = []

    for scheme in selected:

        name = str(scheme).strip().lower()

        if name == "transform":
            name = "syndrome"

        if name not in ("syndrome", "code_offset"):
            raise ValueError(
                "Enrollment schemes must be "
                "'syndrome' or 'code_offset'."
            )

        if name in schemes:
            raise ValueError(
                f"Duplicate enrollment scheme: {name}"
            )

        schemes.append(name)

    if not schemes:
        raise ValueError(
            "At least one enrollment scheme is required."
        )

    return tuple(schemes)


# ============================================================
# 2. Select decoder configurations
# ============================================================

def get_decoder_runs(cfg):

    selected = cfg.decoder

    if isinstance(selected, str):
        selected = (selected,)

    selected = tuple(
        str(name).strip().upper()
        for name in selected
    )

    valid_decoders = {
        "SC",
        "DSCF",
        "SCL",
    }

    if not selected:
        raise ValueError(
            "At least one decoder must be selected."
        )

    if any(
        name not in valid_decoders
        for name in selected
    ):
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
                        "SCL list sizes must be "
                        "positive powers of two."
                    )

                runs.append(("SCL", L))

    return runs


# ============================================================
# 3. Construct polar code
# ============================================================

def construct_polar_code(cfg, n, k):

    construction = str(
        cfg.construction
    ).strip().upper()

    # --------------------------------------------------------
    # 5G construction
    # --------------------------------------------------------

    if construction == "5G":

        frozen, info = generate_5g_ranking(
            k,
            n,
        )

    # --------------------------------------------------------
    # BSC Bhattacharyya construction
    # --------------------------------------------------------

    elif construction in ("BSC", "BHATTACHARYYA"):

        from constructions import bhattacharyya_design

        design_p = float(cfg.design_p)

        if not (0.0 < design_p < 0.5):
            raise ValueError(
                "design_p must lie in (0, 0.5)."
            )

        frozen, info, _ = bhattacharyya_design(
            n,
            k,
            channel="BSC",
            design_p=design_p,
        )

        construction = "BSC"

    # --------------------------------------------------------
    # Custom OURS construction
    # --------------------------------------------------------

    elif construction == "OURS":

        default_path = (
            f"codes/frozen_bits_N{n}_K{k}_our.csv"
        )

        frozen_path = Path(
            getattr(
                cfg,
                "ours_frozen_path",
                None,
            ) or default_path
        )

        if not frozen_path.is_file():
            raise FileNotFoundError(
                f"Custom frozen-bit file not found: "
                f"{frozen_path}"
            )

        raw = np.loadtxt(
            frozen_path,
            delimiter=",",
            ndmin=1,
        ).ravel()

        if (
            not np.all(np.isfinite(raw))
            or not np.all(raw == np.rint(raw))
        ):
            raise ValueError(
                "Custom frozen positions must be "
                "finite integers."
            )

        frozen = raw.astype(np.int32)

        if (
            np.any(frozen < 0)
            or np.any(frozen >= n)
            or len(np.unique(frozen)) != len(frozen)
        ):
            raise ValueError(
                "Custom frozen positions contain "
                "duplicates or invalid indices."
            )

        info = np.setdiff1d(
            np.arange(n),
            frozen,
        )

    else:

        raise ValueError(
            "construction must be 5G, BSC, or OURS."
        )

    # --------------------------------------------------------
    # Sort and validate frozen/information positions
    # --------------------------------------------------------

    frozen = np.sort(
        np.asarray(
            frozen,
            dtype=np.int32,
        )
    )

    info = np.sort(
        np.asarray(
            info,
            dtype=np.int32,
        )
    )

    if (
        len(frozen) != n - k
        or len(info) != k
        or not np.array_equal(
            np.sort(
                np.concatenate(
                    (frozen, info)
                )
            ),
            np.arange(n),
        )
    ):
        raise ValueError(
            "Invalid frozen/information-bit partition."
        )

    return frozen, info, construction


# ============================================================
# 4. CSV utilities
# ============================================================

def create_csv(out_dir, filename):

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = out_dir / filename

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()

    return path


def append_csv(path, row):

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )

        writer.writerow({
            "p": row["p"],
            "BLER": row["BLER"],
            "normalized_complexity":
                row["normalized_complexity"],
        })


# ============================================================
# 5. Verify Monte Carlo results
# ============================================================

def verify_result(
    result,
    decoder,
    L,
    trials,
    nlogn,
    max_blocks,
):

    blocks = int(result["blocks"])

    units = int(
        result["total_sc_equivalent_units"]
    )

    passes = result["total_sc_passes"]

    complexity = float(
        result["normalized_complexity"]
    )

    if not (0 < blocks <= max_blocks):
        raise RuntimeError(
            "Invalid Monte Carlo block count."
        )

    if int(result["nlogn_per_pass"]) != nlogn:
        raise RuntimeError(
            "Incorrect N log2(N) operation count."
        )

    if (
        int(result["total_sc_equivalent_ops"])
        != units * nlogn
    ):
        raise RuntimeError(
            "Inconsistent SC-equivalent operation count."
        )

    if not np.isclose(
        complexity,
        units / blocks,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Incorrect normalized complexity."
        )

    # --------------------------------------------------------
    # SC
    # --------------------------------------------------------

    if decoder == "SC":

        if passes is None:
            raise RuntimeError(
                "Missing SC pass counter."
            )

        if (
            int(passes) != blocks
            or units != blocks
        ):
            raise RuntimeError(
                "SC must execute one pass per frame."
            )

        if complexity != 1.0:
            raise RuntimeError(
                "SC normalized complexity must be 1."
            )

    # --------------------------------------------------------
    # DSCF
    # --------------------------------------------------------

    elif decoder == "DSCF":

        if passes is None:
            raise RuntimeError(
                "Missing DSCF pass counter."
            )

        if int(passes) != units:
            raise RuntimeError(
                "DSCF complexity counter mismatch."
            )

        if not (
            blocks <= units <= blocks * (1 + trials)
        ):
            raise RuntimeError(
                "Invalid DSCF decoding attempts."
            )

    # --------------------------------------------------------
    # SCL
    # --------------------------------------------------------

    elif decoder == "SCL":

        if units != blocks * L:
            raise RuntimeError(
                "Incorrect nominal SCL complexity."
            )

        if not np.isclose(
            complexity,
            float(L),
        ):
            raise RuntimeError(
                "SCL normalized complexity must equal L."
            )

        if result["complexity_model"] != "nominal_L":
            raise RuntimeError(
                "Incorrect SCL complexity model."
            )


# ============================================================
# 6. Verify enrollment before simulation
# ============================================================

def verify_enrollment(
    device,
    frozen,
    info,
):

    """
    Confirm that applying the enrollment offset to
    the enrolled PUF response produces a valid
    zero-frozen polar codeword encoding the key.

    This verification works for both enrollment
    schemes.
    """

    if device.offset is None:
        raise RuntimeError(
            "Enrollment result has no offset. "
            "Update Enrolment.py first."
        )

    x = device.x.astype(np.uint8)

    z = device.offset.astype(np.uint8)

    codeword = device.codeword.astype(np.uint8)

    # Physical response XOR offset = codeword.
    np.testing.assert_array_equal(
        x ^ z,
        codeword,
        err_msg="Enrollment offset identity failed.",
    )

    # Polar transform is self-inverse.
    recovered_u = polar_transform_bits(
        codeword
    )

    # Frozen bits must be zero.
    if not np.all(
        recovered_u[:, frozen] == 0
    ):
        raise RuntimeError(
            "Codeword has nonzero frozen bits."
        )

    # Information bits must equal enrolled key.
    np.testing.assert_array_equal(
        recovered_u[:, info],
        device.key,
        err_msg="Enrolled key consistency check failed.",
    )


# ============================================================
# 7. Main simulation
# ============================================================

def simulate(cfg: SimConfig) -> list[dict]:

    # --------------------------------------------------------
    # Simulation parameters
    # --------------------------------------------------------

    n = 1 << int(cfg.n_log2)

    k = int(cfg.key_length)

    q = float(
        cfg.puf_one_probability
    )

    p_values = tuple(
        float(p) for p in cfg.p_values
    )

    trials = int(cfg.num_trials)

    alpha = float(cfg.alpha)

    min_errors = int(cfg.min_block_errors)

    max_blocks = int(cfg.max_blocks)

    checker_kind = str(
        cfg.checker
    ).strip().lower()

    seed = int(
        getattr(cfg, "seed", 2026)
    )

    scramble = bool(
        getattr(
            cfg,
            "scramble_codeword",
            False,
        )
    )

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else 1
    )

    threads = int(
        getattr(
            cfg,
            "num_threads",
            available_cpus,
        )
    )

    out_dir = Path(cfg.out_dir)

    # --------------------------------------------------------
    # Validate simulation parameters
    # --------------------------------------------------------

    if not (1 <= k < n):
        raise ValueError(
            "Require 0 < K < N."
        )

    if not (0.0 < q < 1.0):
        raise ValueError(
            "puf_one_probability must be in (0, 1)."
        )

    if cfg.all_zero_device and q != 0.5:
        raise ValueError(
            "All-zero baseline requires q=0.5."
        )

    if not p_values or any(
        not (0.0 <= p < 0.5)
        for p in p_values
    ):
        raise ValueError(
            "BSC probabilities must be in [0, 0.5)."
        )

    if threads < 1:
        raise ValueError(
            "num_threads must be positive."
        )

    if trials < 0:
        raise ValueError(
            "num_trials cannot be negative."
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    if min_errors < 1 or max_blocks < 1:
        raise ValueError(
            "Monte Carlo stopping limits "
            "must be positive."
        )

    if checker_kind not in ("hash", "crc"):
        raise ValueError(
            "checker must be 'hash' or 'crc'."
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

    elif checker_kind == "crc":

        if str(cfg.crc_degree).upper() != "CRC16":
            raise ValueError(
                "Native checker supports CRC16 only."
            )

    # --------------------------------------------------------
    # Selected schemes and decoders
    # --------------------------------------------------------

    enrollment_schemes = get_enrollment_schemes(
        cfg
    )

    decoder_runs = get_decoder_runs(
        cfg
    )

    # --------------------------------------------------------
    # Polar-code construction
    # --------------------------------------------------------

    frozen, info, construction = (
        construct_polar_code(
            cfg,
            n,
            k,
        )
    )

    # --------------------------------------------------------
    # Key verification
    # --------------------------------------------------------

    checker = KeyChecker(
        method=checker_kind,
        hash_name=cfg.hash_name,
        hash_bits=int(cfg.hash_bits),
        crc_degree=cfg.crc_degree,
    )

    checker_id = (
        "sha256"
        if checker_kind == "hash"
        else "crc16"
    )

    # --------------------------------------------------------
    # Enrollment object
    # --------------------------------------------------------

    enrollment = SRAMPUFEnrollment(
        n,
        frozen,
        checker=checker,
        biased=(q != 0.5),
        prob_one=q,
    )

    # --------------------------------------------------------
    # Generate ONE fixed physical PUF response.
    #
    # Both schemes use exactly the same X.
    # --------------------------------------------------------

    rng_puf = np.random.default_rng(
        seed
    )

    if cfg.all_zero_device:

        physical_x = np.zeros(
            (1, n),
            dtype=np.uint8,
        )

    else:

        physical_x = None

    syndrome_device = enrollment.enroll(
        batch_size=1,
        x=physical_x,
        rng=rng_puf,
        scheme="syndrome",
    )

    # Save the same physical X for both constructions.
    physical_x = syndrome_device.x.copy()

    # --------------------------------------------------------
    # Independent RNG for conventional code-offset key.
    #
    # The key must not be derived from X.
    # --------------------------------------------------------

    rng_key = np.random.default_rng(
        np.random.SeedSequence(
            [seed, 0xC0DE]
        )
    )

    # --------------------------------------------------------
    # Verify algebraic equivalence for the paired case.
    # --------------------------------------------------------

    _, paired_offset = enrollment.code_offset(
        syndrome_device
    )

    np.testing.assert_array_equal(
        paired_offset,
        syndrome_device.offset,
        err_msg=(
            "Syndrome/code-offset paired "
            "equivalence failed."
        ),
    )

    # --------------------------------------------------------
    # Output directory and filename prefix
    # --------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"BSC_N{n}_K{k}_{construction}_"
        f"{checker_kind}"
    )

    if checker_kind == "hash":
        prefix += str(int(cfg.hash_bits))

    all_rows = []

    # ========================================================
    # 8. Simulation information
    # ========================================================

    print("=" * 100)
    print("AFF3CT POLAR SRAM-PUF MONTE CARLO")
    print("=" * 100)

    print(f"N={n}, K={k}")
    print(f"Construction: {construction}")

    print(
        f"Enrollment schemes: {enrollment_schemes}"
    )

    print(
        f"Decoder configurations: {decoder_runs}"
    )

    print(
        f"DSCF: T={trials}, alpha={alpha}"
    )

    print(f"Checker: {checker_kind}")
    print(f"Threads: {threads}")
    print(f"Scrambling: {scramble}")
    print(f"PUF P(X=1): {q}")
    print(f"Seed: {seed}")

    print(
        "Complexity: actual SC passes for SC/DSCF; "
        "nominal L for SCL."
    )

    print("=" * 100, flush=True)

    # ========================================================
    # 9. Run each enrollment scheme
    # ========================================================

    for scheme in enrollment_schemes:

        # ----------------------------------------------------
        # Enroll with the selected scheme
        # ----------------------------------------------------

        if scheme == "syndrome":

            device = syndrome_device

        elif scheme == "code_offset":

            device = enrollment.enroll(
                x=physical_x,
                rng=rng_key,
                scheme="code_offset",
            )

        else:

            raise RuntimeError(
                f"Unknown enrollment scheme: {scheme}"
            )

        # ----------------------------------------------------
        # Verify this enrollment
        # ----------------------------------------------------

        verify_enrollment(
            device,
            frozen,
            info,
        )

        # ----------------------------------------------------
        # Prepare inputs for native C++ simulation
        # ----------------------------------------------------

        x = device.x[0].astype(
            np.uint8
        ).tolist()

        # IMPORTANT:
        #
        # device.offset works for BOTH schemes.
        #
        # syndrome:
        #   offset = W_full G_N
        #
        # code_offset:
        #   offset = X XOR encode(secret)
        #
        # Do not use device.frozen_full here.
        # It is None for conventional code-offset.

        z = device.offset[0].astype(
            np.uint8
        ).tolist()

        secret = device.key[0].astype(
            np.uint8
        ).tolist()

        public_tag = device.tag[0].astype(
            np.uint8
        ).tolist()

        # ----------------------------------------------------
        # Scheme-specific filename prefix
        # ----------------------------------------------------

        scheme_prefix = (
            f"{prefix}_{scheme.upper()}"
        )

        print()
        print("=" * 100)
        print(f"ENROLLMENT: {scheme.upper()}")
        print("=" * 100)

        print(
            f"Helper bits: {device.helper.shape[1]}"
        )

        print(
            f"Secret bits: {len(secret)}"
        )

        print(
            f"Output prefix: {scheme_prefix}"
        )

        print("=" * 100, flush=True)

        # ====================================================
        # 10. Run each decoder
        # ====================================================

        for decoder_name, L in decoder_runs:

            # ------------------------------------------------
            # Decoder-specific CSV filename
            # ------------------------------------------------

            if decoder_name == "SC":

                filename = (
                    f"{scheme_prefix}_SC.csv"
                )

                label = "SC"

            elif decoder_name == "DSCF":

                filename = (
                    f"{scheme_prefix}_DSCF_T{trials}.csv"
                )

                label = "DSCF"

            else:

                filename = (
                    f"{scheme_prefix}_SCL_L{L}.csv"
                )

                label = f"SCL_L{L}"

            csv_path = create_csv(
                out_dir,
                filename,
            )

            # ------------------------------------------------
            # Initialize native C++ Monte Carlo engine
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Run over crossover probabilities
            # ------------------------------------------------

            for p in p_values:

                # Native C++ performs Monte Carlo
                # and displays live progress.

                result = engine.run(
                    p=p,
                    min_errors=min_errors,
                    max_blocks=max_blocks,
                )

                # --------------------------------------------
                # Verify complexity counters
                # --------------------------------------------

                verify_result(
                    result=result,
                    decoder=decoder_name,
                    L=L,
                    trials=trials,
                    nlogn=n * int(cfg.n_log2),
                    max_blocks=max_blocks,
                )

                # --------------------------------------------
                # CSV: exactly three columns
                # --------------------------------------------

                row = {
                    "p": p,

                    "BLER": float(
                        result["BLER"]
                    ),

                    "normalized_complexity": float(
                        result["normalized_complexity"]
                    ),
                }

                # Write after each completed p.
                append_csv(
                    csv_path,
                    row,
                )

                # --------------------------------------------
                # Retain metadata in memory
                # --------------------------------------------

                all_rows.append({
                    "scheme": scheme,
                    "decoder": label,
                    "list_size": L,
                    **row,
                })

    return all_rows


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    simulate(
        SimConfig()
    )
