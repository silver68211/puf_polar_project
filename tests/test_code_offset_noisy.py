
"""Noisy-channel integration tests for polar SRAM-PUF enrollment.

Run:
    python -m pytest -v test_code_offset_noisy.py

These are bounded regression tests, not precise BLER estimates.
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from config import SimConfig
from Enrolment import SRAMPUFEnrollment
from puf_aff3ct import (
    PolarDSCF,
    PolarSC,
    PolarSCLPUF,
)
from sim_aff3ct import simulate
from utils import (
    generate_5g_ranking,
    polar_transform_bits,
)


# ============================================================
# 1. Noisy native Monte Carlo
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
    checker_kind,
    q,
    scramble,
    p_values,
    tmp_path,
):
    """
    Both enrollment schemes must run under noise and return
    valid BLER and normalized-complexity statistics.
    """

    cfg = replace(
        SimConfig(),
        n_log2=8,
        key_length=128,
        construction="5G",
        puf_one_probability=q,
        all_zero_device=False,
        scramble_codeword=scramble,
        enrollment_schemes=(
            "syndrome",
            "code_offset",
        ),
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

    schemes = (
        "syndrome",
        "code_offset",
    )

    decoders = (
        "SC",
        "DSCF",
        "SCL_L2",
        "SCL_L4",
    )

    expected_count = (
        len(schemes)
        * len(decoders)
        * len(p_values)
    )

    assert len(rows) == expected_count

    # ========================================================
    # Validate Monte Carlo outputs
    # ========================================================

    for row in rows:

        assert row["scheme"] in schemes
        assert row["decoder"] in decoders
        assert row["p"] in p_values

        bler = float(row["BLER"])

        assert np.isfinite(bler)
        assert 0.0 <= bler <= 1.0

        complexity = float(
            row["normalized_complexity"]
        )

        assert np.isfinite(complexity)

        if row["decoder"] == "SC":

            assert complexity == pytest.approx(1.0)

        elif row["decoder"] == "DSCF":

            assert (
                1.0
                <= complexity
                <= 1.0 + cfg.num_trials
            )

        elif row["decoder"] == "SCL_L2":

            assert complexity == pytest.approx(2.0)

        elif row["decoder"] == "SCL_L4":

            assert complexity == pytest.approx(4.0)

    # ========================================================
    # Validate CSV output
    # ========================================================

    csv_paths = sorted(
        tmp_path.glob("*.csv")
    )

    assert len(csv_paths) == (
        len(schemes) * len(decoders)
    )

    for path in csv_paths:

        with path.open(
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            assert reader.fieldnames == [
                "p",
                "BLER",
                "normalized_complexity",
            ]

            csv_rows = list(reader)

        assert len(csv_rows) == len(p_values)

        assert tuple(
            float(r["p"]) for r in csv_rows
        ) == p_values

        for csv_row in csv_rows:

            assert (
                0.0
                <= float(csv_row["BLER"])
                <= 1.0
            )

            assert (
                float(csv_row["normalized_complexity"])
                >= 1.0
            )

    # Do not require independently sampled syndrome and
    # code-offset experiments to have identical measured BLER.
    # Their keys and codewords may differ, and samples are finite.


# ============================================================
# 2. Paired schemes with identical noisy inputs
# ============================================================

@pytest.mark.parametrize("q", [0.5, 0.4])
@pytest.mark.parametrize("scramble", [False, True])
def test_paired_native_decoders_same_noisy_inputs(
    q,
    scramble,
):
    """
    Paired syndrome/code-offset enrollment must produce the
    same decoder decisions for identical input LLRs.

    The code-offset key is deliberately chosen to equal the
    syndrome-derived key.

    This is an algebraic consistency test, not an independent
    code-offset experiment.
    """

    n, k = 256, 128

    frozen, info = generate_5g_ranking(k, n)

    frozen = np.sort(
        np.asarray(frozen, dtype=np.int32)
    )

    info = np.sort(
        np.asarray(info, dtype=np.int32)
    )

    enrollment = SRAMPUFEnrollment(
        n=n,
        frozen_pos=frozen,
        biased=(q != 0.5),
        prob_one=q,
    )

    rng = np.random.default_rng(314159)

    x = (
        rng.random((1, n)) < q
    ).astype(np.uint8)

    # ========================================================
    # Enrollment
    # ========================================================

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

    # ========================================================
    # Optional shared codeword scrambling
    # ========================================================

    if scramble:

        mask = rng.integers(
            0, 2, size=k, dtype=np.uint8
        )

    else:

        mask = np.zeros(
            k, dtype=np.uint8
        )

    u_mask = np.zeros(
        (1, n), dtype=np.uint8
    )

    u_mask[0, info] = mask

    mask_codeword = polar_transform_bits(
        u_mask
    )[0]

    # ========================================================
    # Generate noisy SRAM observations
    # ========================================================

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

    y = np.bitwise_xor(
        x[0][None, :],
        errors,
    )

    # ========================================================
    # Construct LLRs independently for both schemes
    # ========================================================

    def shifted_llrs(offset):

        physical_llrs = (
            (1.0 - 2.0 * y.astype(np.float32))
            * channel_magnitude
            + physical_prior
        )

        shift = (
            offset[0] ^ mask_codeword
        )

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

    np.testing.assert_array_equal(
        llrs_syndrome,
        llrs_paired,
    )

    # ========================================================
    # SC decoding
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
    # Public key-verification callback
    # ========================================================

    expected_tag = syndrome.tag[0]

    def public_verifier(candidate):

        candidate_bits = np.asarray(
            candidate,
            dtype=np.uint8,
        ).reshape(1, -1)

        computed_tag = enrollment.checker.tag(
            candidate_bits
        )[0]

        return bool(
            np.array_equal(
                computed_tag,
                expected_tag,
            )
        )

    # ========================================================
    # DSCF decoding
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
    # SCL decoding, L=4
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

    # ========================================================
    # Compare decoder decisions
    # ========================================================

    for row_a, row_b in zip(
        llrs_syndrome,
        llrs_paired,
    ):

        # --------------------
        # DSCF
        # --------------------

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

        assert (
            result_a["key"]
            == result_b["key"]
        )

        assert (
            result_a["verified"]
            == result_b["verified"]
        )

        assert (
            result_a["attempts"]
            == result_b["attempts"]
        )

        # --------------------
        # SCL
        # --------------------

        result_a = scl_a.decode(
            row_a,
            mask.tolist(),
        )

        result_b = scl_b.decode(
            row_b,
            mask.tolist(),
        )

        assert (
            result_a["key"]
            == result_b["key"]
        )

        assert (
            result_a["verified"]
            == result_b["verified"]
        )

        assert (
            result_a["candidates_checked"]
            == result_b["candidates_checked"]
        )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-v", __file__])
    )
