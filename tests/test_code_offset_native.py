
"""Bounded noiseless native AFF3CT integration tests.

Run:
    python -m pytest -v test_code_offset_native.py
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from config import SimConfig
from puf_aff3ct import (
    BSCMonteCarloPolar,
    PolarDSCF,
    PolarSC,
    PolarSCLPUF,
)
from sim_aff3ct import simulate


# ============================================================
# 1. Native module exports
# ============================================================

def test_native_classes_exported():
    """Verify that all required native classes are available."""

    for cls in (
        BSCMonteCarloPolar,
        PolarSC,
        PolarDSCF,
        PolarSCLPUF,
    ):
        assert callable(cls)


# ============================================================
# 2. Noiseless native reconstruction
# ============================================================

@pytest.mark.parametrize(
    "scheme",
    ["syndrome", "code_offset"],
)
@pytest.mark.parametrize("q", [0.5, 0.4])
@pytest.mark.parametrize("scramble", [False, True])
def test_all_native_decoders_are_noiseless(
    scheme,
    q,
    scramble,
    tmp_path,
):
    """
    At BSC p=0, every decoder must recover the enrolled key.

    Tested decoders:
        SC
        DSCF, T=10
        SCL, L=2
        SCL, L=4
        SCL, L=8
    """

    cfg = replace(
        SimConfig(),
        n_log2=8,
        key_length=128,
        construction="5G",
        enrollment_schemes=(scheme,),
        decoder=("SC", "DSCF", "SCL"),
        list_sizes=(2, 4, 8),
        num_trials=10,
        alpha=0.3367,
        puf_one_probability=q,
        all_zero_device=False,
        scramble_codeword=scramble,
        checker="hash",
        hash_name="sha256",
        hash_bits=128,
        p_values=(0.0,),
        min_block_errors=1,
        max_blocks=32,
        num_threads=1,
        seed=2026,
        out_dir=str(tmp_path),
    )

    cfg.validate()

    rows = simulate(cfg)

    expected = {
        "SC",
        "DSCF",
        "SCL_L2",
        "SCL_L4",
        "SCL_L8",
    }

    # One result per decoder.
    assert len(rows) == len(expected)

    assert {
        row["decoder"] for row in rows
    } == expected

    # ========================================================
    # Check decoder statistics
    # ========================================================

    for row in rows:

        assert row["scheme"] == scheme
        assert float(row["p"]) == 0.0

        # Every decoder must recover the key without noise.
        assert float(row["BLER"]) == 0.0, (
            scheme,
            q,
            scramble,
            row,
        )

        complexity = float(
            row["normalized_complexity"]
        )

        assert np.isfinite(complexity)

        if row["decoder"] == "SC":

            assert complexity == pytest.approx(1.0)

        elif row["decoder"].startswith("SCL_L"):

            list_size = int(row["decoder"][5:])

            assert complexity == pytest.approx(
                list_size
            )

        else:

            # DSCF can use between 1 and 1+T attempts.
            assert 1.0 <= complexity <= 11.0

    # ========================================================
    # Check CSV output files
    # ========================================================

    files = sorted(
        tmp_path.glob("*.csv")
    )

    assert len(files) == len(expected)

    for file in files:

        with file.open(
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            assert reader.fieldnames == [
                "p",
                "BLER",
                "normalized_complexity",
            ]

            records = list(reader)

        assert len(records) == 1

        assert float(records[0]["p"]) == 0.0
        assert float(records[0]["BLER"]) == 0.0


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-v", __file__])
    )
