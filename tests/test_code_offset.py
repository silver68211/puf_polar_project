
"""Algebraic tests for syndrome and code-offset SRAM-PUF enrollment.

Run:
    python -m pytest -v test_code_offset.py
"""

from __future__ import annotations

import numpy as np
import pytest

from Enrolment import SRAMPUFEnrollment
from utils import polar_transform_bits


def make_enrollment(n=64, k=16, q=0.5):
    """Create an enrollment object with a fixed information set."""
    frozen = np.arange(n - k, dtype=np.int32)

    return SRAMPUFEnrollment(
        n=n,
        frozen_pos=frozen,
        biased=(q != 0.5),
        prob_one=q,
    )


def bits(rng, shape):
    return rng.integers(
        0, 2, size=shape, dtype=np.uint8
    )


# ============================================================
# 1. Polar-transform tests
# ============================================================

@pytest.mark.parametrize(
    "n,k",
    [(8, 4), (64, 16), (256, 128)],
)
def test_polar_transform_is_involution(n, k):
    """The polar transform is self-inverse over GF(2)."""

    rng = np.random.default_rng(1234 + n)
    x = bits(rng, (5, n))

    recovered = polar_transform_bits(
        polar_transform_bits(x)
    )

    np.testing.assert_array_equal(recovered, x)


@pytest.mark.parametrize(
    "n,k",
    [(8, 4), (64, 16), (256, 128)],
)
def test_polar_transform_is_linear(n, k):
    """Verify (a XOR b) G = aG XOR bG."""

    rng = np.random.default_rng(2345 + n)
    a, b = bits(rng, (2, n))

    lhs = polar_transform_bits(
        np.bitwise_xor(a[None], b[None])
    )

    rhs = np.bitwise_xor(
        polar_transform_bits(a[None]),
        polar_transform_bits(b[None]),
    )

    np.testing.assert_array_equal(lhs, rhs)


# ============================================================
# 2. Syndrome enrollment
# ============================================================

@pytest.mark.parametrize("q", [0.5, 0.4])
@pytest.mark.parametrize(
    "n,k",
    [(64, 16), (256, 128)],
)
def test_syndrome_enrollment_identity(n, k, q):
    """Check key/helper splitting and offset algebra."""

    rng = np.random.default_rng(1200 + n)

    enrollment = make_enrollment(n, k, q)

    x = (
        rng.random((3, n)) < q
    ).astype(np.uint8)

    record = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    u = polar_transform_bits(x)

    np.testing.assert_array_equal(record.x, x)
    np.testing.assert_array_equal(record.c, u)

    np.testing.assert_array_equal(
        record.key,
        u[:, enrollment.info_pos],
    )

    np.testing.assert_array_equal(
        record.helper,
        u[:, enrollment.frozen_pos],
    )

    assert record.key.shape == (3, k)
    assert record.helper.shape == (3, n - k)

    # Remove the syndrome-derived offset.
    recovered_codeword = np.bitwise_xor(
        x,
        record.offset,
    )

    recovered_u = polar_transform_bits(
        recovered_codeword
    )

    # All frozen positions must become zero.
    np.testing.assert_array_equal(
        recovered_u[:, enrollment.frozen_pos],
        0,
    )

    # Information positions must contain the enrolled key.
    np.testing.assert_array_equal(
        recovered_u[:, enrollment.info_pos],
        record.key,
    )


# ============================================================
# 3. Paired syndrome/code-offset equivalence
# ============================================================

@pytest.mark.parametrize("q", [0.5, 0.4])
def test_paired_code_offset_matches_syndrome(q):
    """Same physical X and same key must yield identical offsets."""

    n, k = 256, 128

    rng = np.random.default_rng(77)

    enrollment = make_enrollment(n, k, q)

    x = (
        rng.random((4, n)) < q
    ).astype(np.uint8)

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
        paired.key,
        syndrome.key,
    )

    np.testing.assert_array_equal(
        paired.offset,
        syndrome.offset,
    )

    np.testing.assert_array_equal(
        paired.tag,
        syndrome.tag,
    )

    encoded = np.bitwise_xor(
        x,
        paired.offset,
    )

    u = polar_transform_bits(encoded)

    np.testing.assert_array_equal(
        u[:, enrollment.frozen_pos],
        0,
    )

    np.testing.assert_array_equal(
        u[:, enrollment.info_pos],
        syndrome.key,
    )


# ============================================================
# 4. Independent code-offset enrollment
# ============================================================

def test_independent_code_offset_uses_specified_key():
    """Independent code offset must preserve the chosen key."""

    n, k = 64, 16

    rng = np.random.default_rng(99)

    enrollment = make_enrollment(n, k)

    x = bits(rng, (3, n))
    chosen_key = bits(rng, (3, k))

    result = enrollment.enroll(
        x=x,
        scheme="code_offset",
        key=chosen_key,
    )

    np.testing.assert_array_equal(
        result.key,
        chosen_key,
    )

    u = polar_transform_bits(
        np.bitwise_xor(x, result.offset)
    )

    np.testing.assert_array_equal(
        u[:, enrollment.frozen_pos],
        0,
    )

    np.testing.assert_array_equal(
        u[:, enrollment.info_pos],
        chosen_key,
    )


# ============================================================
# 5. Zero-helper case
# ============================================================

def test_syndrome_offset_is_zero_when_helper_is_zero():
    """Zero frozen bits imply a zero syndrome-derived offset."""

    n, k = 64, 16

    rng = np.random.default_rng(33)

    enrollment = make_enrollment(n, k)

    u = np.zeros((2, n), dtype=np.uint8)

    u[:, enrollment.info_pos] = bits(
        rng,
        (2, k),
    )

    x = polar_transform_bits(u)

    result = enrollment.enroll(
        x=x,
        scheme="syndrome",
    )

    np.testing.assert_array_equal(
        result.helper,
        0,
    )

    np.testing.assert_array_equal(
        result.offset,
        0,
    )


# ============================================================
# 6. Invalid arguments
# ============================================================

@pytest.mark.parametrize(
    "bad_n",
    [0, 3, 15, 65],
)
def test_invalid_block_length_rejected(bad_n):
    """Reject lengths that are not positive powers of two."""

    with pytest.raises((ValueError, TypeError)):
        SRAMPUFEnrollment(
            n=bad_n,
            frozen_pos=[],
        )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-v", __file__])
    )
