
"""Noisy-channel integration tests for polar SRAM-PUF enrollment.

Run:
    python -m pytest -v test_code_offset_noisy.py
    python test_code_offset_noisy.py

These are bounded regression tests, not BLER performance estimates.
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from config import SimConfig
from Enrolment import SRAMPUFEnrollment
from puf_aff3ct import PolarDSCF, PolarSC, PolarSCLPUF
from sim_aff3ct import simulate
from utils import generate_5g_ranking, polar_transform_bits


# ============================================================
# 1. End-to-end native Monte Carlo at p > 0
# ============================================================

@pytest.mark.parametrize(
    "checker_kind,q,scramble,p_values",
    [
        ("hash", 0.5, False, (0.05, 0.10, 0.15)),
        ("hash", 0.4, True, (0.05, 0.10, 0.15)),
        ("crc", 0.5, True, (0.10,)),
    ],
)
def test_noisy_native_monte_carlo(
    checker_kind, q, scramble, p_values, tmp_path
):
    """Both schemes must run without errors and report valid statistics."""

    cfg = replace(
        SimConfig(),
        n_log2=8,
        key_length=128,
        construction="5G",
        puf_one_probability=q,
        all_zero_device=False,
        scramble_codeword=scramble,
        enrollment_schemes=("syndrome", "code_offset"),
        decoder=("SC", "DSCF", "SCL"),
        list_sizes=(2, 4),
        num_trials=10,
        alpha=0.3367,
        checker=checker_kind,
        hash_name="sha256",
        hash_bits=128,
        crc_degree="CRC16",
        p_values=p_values,
        min_block_errors=8,
        max_blocks=64,
        num_threads=1,
        seed=2026,
        out_dir=str(tmp_path),
    )

    rows = simulate(cfg)
    schemes = ("syndrome", "code_offset")
    decoders = ("SC", "DSCF", "SCL_L2", "SCL_L4")

    # Two schemes x four decoder settings x selected p values.
    assert len(rows) == len(schemes) * len(decoders) * len(p_values)

    for row in rows:
        assert row["scheme"] in schemes
        assert row["decoder"] in decoders
        assert row["p"] in p_values
        assert np.isfinite(row["BLER"])
        assert 0.0 <= row["BLER"] <= 1.0

        complexity = row["normalized_complexity"]
        assert np.isfinite(complexity)
        if row["decoder"] == "SC":
            assert complexity == 1.0
        elif row["decoder"] == "DSCF":
            assert 1.0 <= complexity <= 1.0 + cfg.num_trials
        elif row["decoder"] == "SCL_L2":
            assert complexity == 2.0
        else:
            assert complexity == 4.0

    # Confirm each scheme/decoder has its own CSV with all p values.
    csv_paths = sorted(tmp_path.glob("*.csv"))
    assert len(csv_paths) == len(schemes) * len(decoders)

    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "p", "BLER", "normalized_complexity"
            ]
            csv_rows = list(reader)

        assert len(csv_rows) == len(p_values)
        assert tuple(float(r["p"]) for r in csv_rows) == p_values
        for csv_row in csv_rows:
            assert 0.0 <= float(csv_row["BLER"]) <= 1.0
            assert float(csv_row["normalized_complexity"]) >= 1.0

    # Do not assert equal BLER between independently generated keys:
    # these are different codewords, with finite Monte Carlo samples.


# ============================================================
# 2. Exact paired decoding with the same key AND noise
# ============================================================

@pytest.mark.parametrize("q", [0.5, 0.4])
@pytest.mark.parametrize("scramble", [False, True])
def test_paired_native_decoders_same_noisy_inputs(q, scramble):
    """Paired schemes must give the same native decisions for identical LLRs.

    Here the code-offset scheme is intentionally enrolled using the
    syndrome-derived key. This checks algebraic/decoder equivalence,
    NOT the independent-key code-offset experiment.
    """

    n, k = 256, 128
    frozen, info = generate_5g_ranking(k, n)
    frozen = np.sort(np.asarray(frozen, dtype=np.int32))
    info = np.sort(np.asarray(info, dtype=np.int32))

    enrollment = SRAMPUFEnrollment(
        n=n,
        frozen_pos=frozen,
        biased=(q != 0.5),
        prob_one=q,
    )

    rng = np.random.default_rng(314159)
    x = (rng.random((1, n)) < q).astype(np.uint8)

    syndrome = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    paired = enrollment.enroll(
        x=x,
        scheme="code_offset",
        key=syndrome.key,
    )

    np.testing.assert_array_equal(
        syndrome.offset,
        paired.offset,
    )

    np.testing.assert_array_equal(
        syndrome.tag,
        paired.tag,
    )

    # --------------------------------------------------------
    # Optional random polar-codeword scrambling
    # --------------------------------------------------------

    mask = (
        rng.integers(
            0, 2, size=k, dtype=np.uint8
        )
        if scramble
        else np.zeros(k, dtype=np.uint8)
    )

    u_mask = np.zeros(
        (1, n),
        dtype=np.uint8,
    )

    u_mask[0, info] = mask

    mask_codeword = polar_transform_bits(
        u_mask
    )[0]

    # --------------------------------------------------------
    # Generate the SAME BSC errors for both schemes
    # --------------------------------------------------------

    p = 0.10

    channel_magnitude = np.log(
        (1.0 - p) / p
    )

    physical_prior = np.log(
        (1.0 - q) / q
    )

    errors = (
        rng.random((8, n)) < p
    ).astype(np.uint8)

    y = x[0][None, :] ^ errors

    # --------------------------------------------------------
    # Construct channel LLRs
    # --------------------------------------------------------

    def shifted_llrs(offset):
        physical_llrs = (
            (
                1.0 - 2.0 * y.astype(np.float32)
            ) * channel_magnitude
            + physical_prior
        )

        shift = offset[0] ^ mask_codeword

        return np.ascontiguousarray(
            np.where(
                shift[None, :] == 1,
                -physical_llrs,
                physical_llrs,
            ),
            dtype=np.float32,
        )

    llrs_syndrome = shifted_llrs(
        syndrome.offset
    )

    llrs_paired = shifted_llrs(
        paired.offset
    )

    # Identical key and offset must give identical LLRs.
    np.testing.assert_array_equal(
        llrs_syndrome,
        llrs_paired,
    )

    # ========================================================
    # 3. Paired SC decoding
    # ========================================================

    sc_a = PolarSC(
        n,
        frozen.tolist(),
    )

    sc_b = PolarSC(
        n,
        frozen.tolist(),
    )

    sc_keys_a = (
        sc_a.decode(llrs_syndrome)
        ^ mask[None, :]
    )

    sc_keys_b = (
        sc_b.decode(llrs_paired)
        ^ mask[None, :]
    )

    np.testing.assert_array_equal(
        sc_keys_a,
        sc_keys_b,
    )

    # ========================================================
    # 4. Public verification callback for DSCF
    # ========================================================

    expected_tag = syndrome.tag[0]

    def public_verifier(candidate):
        bits = np.asarray(
            candidate,
            dtype=np.uint8,
        ).reshape(1, -1)

        candidate_tag = enrollment.checker.tag(
            bits
        )[0]

        return bool(
            np.array_equal(
                candidate_tag,
                expected_tag,
            )
        )

    # ========================================================
    # 5. Paired DSCF decoding
    # ========================================================

    dscf_a = PolarDSCF(
        n,
        frozen.tolist(),
        0.3367,
        10,
    )

    dscf_b = PolarDSCF(
        n,
        frozen.tolist(),
        0.3367,
        10,
    )

    # ========================================================
    # 6. Paired SCL decoding
    # ========================================================

    scl_a = PolarSCLPUF(
        n,
        frozen.tolist(),
        4,
        expected_tag.tolist(),
        "sha256",
        128,
    )

    scl_b = PolarSCLPUF(
        n,
        frozen.tolist(),
        4,
        expected_tag.tolist(),
        "sha256",
        128,
    )

    # --------------------------------------------------------
    # Compare decoded keys for each noisy observation
    # --------------------------------------------------------

    for row_a, row_b in zip(
        llrs_syndrome,
        llrs_paired,
    ):

        # ----------------------------------------------------
        # DSCF
        # ----------------------------------------------------

        result_a = dscf_a.decode(
            row_a,
            public_verifier,
            mask.tolist(),
        )

        result_b = dscf_b.decode(
            row_b,
            public_verifier,
            mask.tolist(),
        )

        assert result_a["key"] == result_b["key"]

        assert (
            result_a["verified"]
            == result_b["verified"]
        )

        assert (
            result_a["attempts"]
            == result_b["attempts"]
        )

        # ----------------------------------------------------
        # SCL
        # ----------------------------------------------------

        result_a = scl_a.decode(
            row_a,
            mask.tolist(),
        )

        result_b = scl_b.decode(
            row_b,
            mask.tolist(),
        )

        assert result_a["key"] == result_b["key"]

        assert (
            result_a["verified"]
            == result_b["verified"]
        )

        assert (
            result_a["candidates_checked"]
            == result_b["candidates_checked"]
        )


# ============================================================
# 7. Run tests
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            "-v",
            __file__,
        ])
    )
