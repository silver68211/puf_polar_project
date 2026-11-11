
"""
Validation tests for SRAM-PUF polar enrollment.

Schemes:
    1. Syndrome / transform enrollment
    2. Conventional code-offset enrollment

Tests:
    - Correct helper-data dimensions
    - Correct polar encoding
    - Noiseless key reconstruction
    - Syndrome and paired code-offset equivalence
    - Independent random-key enrollment
    - Externally supplied secret keys
    - Correct BSC offset transformation
    - Public verification tags
    - Biased PUF responses
    - Invalid input handling

Run:
    python test_code_offset.py

Or:
    pytest -v test_code_offset.py
"""

import numpy as np
import pytest

from Enrolment import SRAMPUFEnrollment
from utils import polar_transform_bits


# ============================================================
# Configuration
# ============================================================

N = 256
K = 128
SEED = 2026

# Use a deterministic information/frozen partition.
# These tests validate enrollment algebra, not code performance.
FROZEN = np.arange(N - K, dtype=np.int32)

INFO = np.setdiff1d(
    np.arange(N),
    FROZEN,
)


def make_enrollment(prob_one=0.5):

    return SRAMPUFEnrollment(
        n=N,
        frozen_pos=FROZEN,
        biased=(prob_one != 0.5),
        prob_one=prob_one,
    )


def make_puf(batch_size=4, prob_one=0.5):

    rng = np.random.default_rng(SEED)

    return (
        rng.random((batch_size, N)) < prob_one
    ).astype(np.uint8)


# ============================================================
# 1. Syndrome enrollment
# ============================================================

def test_syndrome_enrollment():

    enrollment = make_enrollment()

    x = make_puf()

    record = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    assert record.scheme == "syndrome"

    assert record.x.shape == (4, N)
    assert record.c.shape == (4, N)
    assert record.key.shape == (4, K)
    assert record.helper.shape == (4, N - K)

    assert record.frozen_full.shape == (4, N)
    assert record.offset.shape == (4, N)
    assert record.codeword.shape == (4, N)

    # C = X G_N
    expected_c = polar_transform_bits(x)

    np.testing.assert_array_equal(
        record.c,
        expected_c,
    )

    # S = C_A
    np.testing.assert_array_equal(
        record.key,
        expected_c[:, INFO],
    )

    # W = C_F
    np.testing.assert_array_equal(
        record.helper,
        expected_c[:, FROZEN],
    )

    # Z = W_full G_N
    expected_offset = polar_transform_bits(
        record.frozen_full
    )

    np.testing.assert_array_equal(
        record.offset,
        expected_offset,
    )


# ============================================================
# 2. Code-offset enrollment
# ============================================================

def test_code_offset_enrollment():

    enrollment = make_enrollment()

    x = make_puf()

    record = enrollment.enroll(
        x=x,
        scheme="code_offset",
        rng=np.random.default_rng(SEED + 1),
    )

    assert record.scheme == "code_offset"

    assert record.key.shape == (4, K)
    assert record.helper.shape == (4, N)
    assert record.offset.shape == (4, N)
    assert record.codeword.shape == (4, N)

    assert record.frozen_full is None

    # Conventional code-offset identity:
    # W_offset = X XOR encode(S)

    np.testing.assert_array_equal(
        record.helper,
        record.x ^ record.codeword,
    )

    np.testing.assert_array_equal(
        record.offset,
        record.helper,
    )

    # Encoded codeword must have zero frozen bits.
    u = polar_transform_bits(
        record.codeword
    )

    assert np.all(
        u[:, FROZEN] == 0
    )

    np.testing.assert_array_equal(
        u[:, INFO],
        record.key,
    )


# ============================================================
# 3. Algebraic equivalence of the two schemes
# ============================================================

def test_paired_code_offset_equivalence():

    enrollment = make_enrollment()

    x = make_puf()

    # Current syndrome scheme.
    syndrome = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    # Construct conventional code offset
    # using the SAME source-derived key.

    codeword, offset = enrollment.code_offset(
        syndrome
    )

    # Both schemes must produce exactly
    # the same full-length reconstruction offset.

    np.testing.assert_array_equal(
        offset,
        syndrome.offset,
    )

    np.testing.assert_array_equal(
        codeword,
        syndrome.codeword,
    )

    # Independently construct a code-offset
    # enrollment with the identical key.

    paired = enrollment.enroll(
        x=x,
        scheme="code_offset",
        key=syndrome.key,
    )

    np.testing.assert_array_equal(
        paired.key,
        syndrome.key,
    )

    np.testing.assert_array_equal(
        paired.offset,
        syndrome.offset,
    )

    np.testing.assert_array_equal(
        paired.codeword,
        syndrome.codeword,
    )

    np.testing.assert_array_equal(
        paired.tag,
        syndrome.tag,
    )


# ============================================================
# 4. Noiseless key reconstruction
# ============================================================

@pytest.mark.parametrize(
    "scheme",
    ["syndrome", "code_offset"],
)
def test_noiseless_reconstruction(scheme):

    enrollment = make_enrollment()

    record = enrollment.enroll(
        x=make_puf(),
        scheme=scheme,
        rng=np.random.default_rng(SEED + 2),
    )

    # The input to the zero-frozen polar decoder.
    corrected = (
        record.x ^ record.offset
    )

    np.testing.assert_array_equal(
        corrected,
        record.codeword,
    )

    # Recover the polar-domain vector.
    u_hat = polar_transform_bits(corrected)

    # Check frozen positions.
    assert np.all(
        u_hat[:, FROZEN] == 0
    )

    # Reconstruct the enrolled key.
    key_hat = u_hat[:, INFO]

    np.testing.assert_array_equal(
        key_hat,
        record.key,
    )


# ============================================================
# 5. Independent random key generation
# ============================================================

def test_independent_key_generation():

    enrollment = make_enrollment()

    x = make_puf()

    # Separate RNG streams for the same input.
    record1 = enrollment.enroll(
        x=x,
        scheme="code_offset",
        rng=np.random.default_rng(123),
    )

    record2 = enrollment.enroll(
        x=x,
        scheme="code_offset",
        rng=np.random.default_rng(456),
    )

    # Independently generated keys should differ
    # for these deterministic test seeds.

    assert not np.array_equal(
        record1.key,
        record2.key,
    )

    # Different keys imply different codewords.
    assert not np.array_equal(
        record1.codeword,
        record2.codeword,
    )

    # Both must be valid enrollments.
    for record in (record1, record2):

        np.testing.assert_array_equal(
            record.x ^ record.offset,
            record.codeword,
        )


# ============================================================
# 6. Externally supplied secret
# ============================================================

def test_supplied_key():

    enrollment = make_enrollment()

    x = make_puf(batch_size=2)

    rng = np.random.default_rng(789)

    secret = rng.integers(
        0,
        2,
        size=(2, K),
        dtype=np.uint8,
    )

    record = enrollment.enroll(
        x=x,
        scheme="code_offset",
        key=secret,
    )

    np.testing.assert_array_equal(
        record.key,
        secret,
    )

    expected_codeword = enrollment.encode_key(
        secret
    )

    np.testing.assert_array_equal(
        record.codeword,
        expected_codeword,
    )


# ============================================================
# 7. BSC channel transformation
# ============================================================

@pytest.mark.parametrize(
    "scheme",
    ["syndrome", "code_offset"],
)
def test_bsc_transformation(scheme):

    enrollment = make_enrollment()

    x = make_puf(batch_size=2)

    record = enrollment.enroll(
        x=x,
        scheme=scheme,
        rng=np.random.default_rng(SEED + 3),
    )

    # Generate independent BSC errors.
    p = 0.08

    rng = np.random.default_rng(999)

    errors = (
        rng.random(x.shape) < p
    ).astype(np.uint8)

    # Noisy SRAM-PUF response.
    y = x ^ errors

    # Actual input to polar decoding.
    decoder_input = y ^ record.offset

    # Expected zero-frozen codeword through BSC.
    expected = record.codeword ^ errors

    np.testing.assert_array_equal(
        decoder_input,
        expected,
    )


# ============================================================
# 8. Identical noisy decoder input for paired schemes
# ============================================================

def test_paired_noisy_inputs():

    enrollment = make_enrollment()

    x = make_puf(batch_size=3)

    syndrome = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    paired = enrollment.enroll(
        x=x,
        scheme="code_offset",
        key=syndrome.key,
    )

    rng = np.random.default_rng(111)

    errors = (
        rng.random(x.shape) < 0.1
    ).astype(np.uint8)

    y = x ^ errors

    decoder_input_syndrome = (
        y ^ syndrome.offset
    )

    decoder_input_offset = (
        y ^ paired.offset
    )

    np.testing.assert_array_equal(
        decoder_input_syndrome,
        decoder_input_offset,
    )


# ============================================================
# 9. Public verification tags
# ============================================================

@pytest.mark.parametrize(
    "scheme",
    ["syndrome", "code_offset"],
)
def test_public_tags(scheme):

    enrollment = make_enrollment()

    record = enrollment.enroll(
        x=make_puf(),
        scheme=scheme,
        rng=np.random.default_rng(SEED + 4),
    )

    expected_tag = enrollment.checker.tag(
        record.key
    )

    np.testing.assert_array_equal(
        record.tag,
        expected_tag,
    )


# ============================================================
# 10. Biased SRAM-PUF responses
# ============================================================

@pytest.mark.parametrize(
    "scheme",
    ["syndrome", "code_offset"],
)
def test_biased_puf(scheme):

    enrollment = make_enrollment(
        prob_one=0.3
    )

    x = make_puf(
        batch_size=4,
        prob_one=0.3,
    )

    record = enrollment.enroll(
        x=x,
        scheme=scheme,
        rng=np.random.default_rng(SEED + 5),
    )

    np.testing.assert_array_equal(
        record.x ^ record.offset,
        record.codeword,
    )

    decoded_u = polar_transform_bits(
        record.codeword
    )

    np.testing.assert_array_equal(
        decoded_u[:, INFO],
        record.key,
    )

    assert np.all(
        decoded_u[:, FROZEN] == 0
    )


# ============================================================
# 11. Invalid inputs
# ============================================================

def test_invalid_inputs():

    enrollment = make_enrollment()

    x = make_puf(batch_size=2)

    # Invalid enrollment scheme.
    with pytest.raises(ValueError):

        enrollment.enroll(
            x=x,
            scheme="invalid",
        )

    # Incorrect key length.
    with pytest.raises(ValueError):

        enrollment.enroll(
            x=x,
            scheme="code_offset",
            key=np.zeros(
                (2, K - 1),
                dtype=np.uint8,
            ),
        )

    # Incorrect key batch size.
    with pytest.raises(ValueError):

        enrollment.enroll(
            x=x,
            scheme="code_offset",
            key=np.zeros(
                (3, K),
                dtype=np.uint8,
            ),
        )

    # Supplying an external key to the
    # syndrome scheme is not permitted.
    with pytest.raises(ValueError):

        enrollment.enroll(
            x=x,
            scheme="syndrome",
            key=np.zeros(
                (2, K),
                dtype=np.uint8,
            ),
        )

    # Nonbinary physical response.
    invalid_x = x.copy()
    invalid_x[0, 0] = 2

    with pytest.raises(ValueError):

        enrollment.enroll(
            x=invalid_x,
            scheme="code_offset",
        )


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    import pytest as _pytest

    raise SystemExit(
        _pytest.main([
            "-v",
            __file__,
        ])
    )
