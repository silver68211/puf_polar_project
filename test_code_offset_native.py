
"""
Native AFF3CT integration tests for SRAM-PUF enrollment.

Tests both:
    1. Syndrome enrollment
    2. Conventional code-offset enrollment

Decoders:
    SC
    DSCF
    SCL (L=2,4)

Checks:
    - Noiseless reconstruction (BLER=0)
    - Unbiased and biased SRAM-PUFs
    - Scrambling enabled and disabled
    - Native Monte Carlo result consistency
    - Correct scheme-specific CSV files

Run:
    python test_code_offset_native.py

Or:
    python -m pytest -v test_code_offset_native.py
"""

from dataclasses import replace

import pandas as pd
import pytest

from config import SimConfig
from sim_aff3ct import simulate


# ============================================================
# 1. Test native decoding for both enrollment schemes
# ============================================================

@pytest.mark.parametrize(
    "prob_one",
    [0.5, 0.4],
)
@pytest.mark.parametrize(
    "scramble",
    [False, True],
)
def test_noiseless_native_decoding(
    prob_one,
    scramble,
    tmp_path,
):
    """
    At p=0, every decoder must reconstruct
    the enrolled key without errors.
    """

    cfg = replace(
        SimConfig(),

        n_log2=8,
        key_length=128,

        construction="5G",

        puf_one_probability=prob_one,

        all_zero_device=False,

        scramble_codeword=scramble,

        enrollment_schemes=(
            "syndrome",
            "code_offset",
        ),

        decoder=(
            "SC",
            "DSCF",
            "SCL",
        ),

        list_sizes=(2, 4),

        num_trials=10,
        alpha=0.3367,

        checker="hash",
        hash_name="sha256",
        hash_bits=128,

        p_values=(0.0,),

        min_block_errors=1,
        max_blocks=16,

        num_threads=1,

        seed=2026,

        out_dir=str(tmp_path),
    )

    # --------------------------------------------------------
    # Execute native Monte Carlo
    # --------------------------------------------------------

    results = simulate(cfg)

    # Two schemes × four decoder configurations.
    assert len(results) == 8

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    for row in results:

        assert row["scheme"] in (
            "syndrome",
            "code_offset",
        )

        assert row["decoder"] in (
            "SC",
            "DSCF",
            "SCL_L2",
            "SCL_L4",
        )

        assert row["p"] == 0.0

        assert row["BLER"] == 0.0, (
            f"Native reconstruction failed: {row}"
        )

        assert row["normalized_complexity"] >= 1.0

    # --------------------------------------------------------
    # Verify CSV output
    # --------------------------------------------------------

    csv_files = list(
        tmp_path.glob("*.csv")
    )

    assert len(csv_files) == 8

    for path in csv_files:

        df = pd.read_csv(path)

        assert list(df.columns) == [
            "p",
            "BLER",
            "normalized_complexity",
        ]

        assert len(df) == 1
        assert df["p"].iloc[0] == 0.0
        assert df["BLER"].iloc[0] == 0.0


# ============================================================
# 2. Check correctness of complexity reporting
# ============================================================

def test_native_complexity(tmp_path):

    cfg = replace(
        SimConfig(),

        n_log2=8,
        key_length=128,

        puf_one_probability=0.5,

        enrollment_schemes=(
            "syndrome",
            "code_offset",
        ),

        decoder=(
            "SC",
            "DSCF",
            "SCL",
        ),

        list_sizes=(2, 4),

        scramble_codeword=False,

        p_values=(0.0,),

        min_block_errors=1,
        max_blocks=16,

        num_threads=1,

        out_dir=str(tmp_path),
    )

    results = simulate(cfg)

    for row in results:

        decoder = row["decoder"]
        complexity = row["normalized_complexity"]

        if decoder == "SC":
            assert complexity == 1.0

        elif decoder == "DSCF":
            assert complexity == 1.0

        elif decoder == "SCL_L2":
            assert complexity == 2.0

        elif decoder == "SCL_L4":
            assert complexity == 4.0


# ============================================================
# 3. Standalone execution
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        pytest.main([
            "-v",
            __file__,
        ])
    )
