
"""
SRAM-PUF enrollment using polar codes.

Supported enrollment schemes
----------------------------
1. Syndrome / transform construction:
   Based on Chen et al. (2017).

   U = X G_N
   S = U_A
   W = U_F

   The public helper data contains N-K bits.

2. Conventional code-offset construction:

   S is generated independently of the PUF response.
   C = encode(S), with frozen bits equal to zero.
   W_offset = X XOR C

   The public helper data contains N bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import sample_puf_bits, polar_transform_bits
from keycheck import KeyChecker


# ============================================================
# 1. Enrollment result
# ============================================================

@dataclass
class EnrollmentResult:
    x: np.ndarray
    # [B,N] Original physical SRAM-PUF response.

    c: np.ndarray
    # [B,N] Polar transform of the physical response:
    # c = X G_N, for both enrollment schemes.

    helper: np.ndarray
    # Public helper data:
    # syndrome:    [B,N-K]
    # code_offset: [B,N]

    key: np.ndarray
    # [B,K] Enrolled secret.
    #
    # syndrome:    PUF-derived
    # code_offset: independently generated or supplied

    tag: np.ndarray
    # [B,T] Public SHA-256 digest bytes or CRC bits.

    frozen_full: np.ndarray | None
    # syndrome:
    #   [B,N] Frozen helper bits embedded in U.
    #
    # code_offset:
    #   None, because the full code offset is
    #   not a frozen-bit vector.

    scheme: str = "syndrome"

    codeword: np.ndarray | None = None
    # [B,N] Valid zero-frozen polar codeword
    # encoding the enrolled key.

    offset: np.ndarray | None = None
    # [B,N] Full reconstruction offset:
    #
    # offset = X XOR codeword
    #
    # This is the vector applied to the noisy PUF
    # response before polar decoding.


# ============================================================
# 2. SRAM-PUF enrollment
# ============================================================

class SRAMPUFEnrollment:

    def __init__(
        self,
        n: int,
        frozen_pos,
        *,
        biased: bool = False,
        prob_one: float = 0.4,
        checker: KeyChecker | None = None,
    ):

        # ----------------------------------------------------
        # Validate code length
        # ----------------------------------------------------

        if n < 2 or n & (n - 1):
            raise ValueError(
                "n must be a power of two."
            )

        positions = np.asarray(
            frozen_pos,
            dtype=np.int32,
        )

        if (
            positions.ndim != 1
            or len(np.unique(positions)) != len(positions)
        ):
            raise ValueError(
                "frozen_pos must contain unique indices."
            )

        if np.any(positions < 0) or np.any(positions >= n):
            raise ValueError(
                "Frozen index out of bounds."
            )

        # ----------------------------------------------------
        # Polar-code parameters
        # ----------------------------------------------------

        self.n = int(n)

        self.frozen_pos = np.sort(
            positions
        )

        self.info_pos = np.setdiff1d(
            np.arange(n),
            self.frozen_pos,
        )

        self.k = len(self.info_pos)

        if not (0 < self.k < self.n):
            raise ValueError(
                "Require 0 < K < N."
            )

        # ----------------------------------------------------
        # PUF parameters
        # ----------------------------------------------------

        self.biased = bool(biased)

        self.prob_one = (
            float(prob_one)
            if biased
            else 0.5
        )

        if not (0 < self.prob_one < 1):
            raise ValueError(
                "prob_one must be in (0,1)."
            )

        # ----------------------------------------------------
        # Key verification
        # ----------------------------------------------------

        self.checker = (
            KeyChecker("hash")
            if checker is None
            else checker
        )

    # ========================================================
    # 3. Validate binary arrays
    # ========================================================

    @staticmethod
    def _validate_bits(
        values,
        width: int,
        name: str,
    ) -> np.ndarray:

        values = np.asarray(values)

        if values.ndim == 1:
            values = values[None, :]

        if (
            values.ndim != 2
            or values.shape[1] != width
            or values.shape[0] < 1
        ):
            raise ValueError(
                f"{name} must have shape [B,{width}]."
            )

        if not np.all(
            (values == 0) | (values == 1)
        ):
            raise ValueError(
                f"{name} must contain binary values."
            )

        return values.astype(
            np.uint8,
            copy=True,
        )

    # ========================================================
    # 4. Polar encoding with zero frozen bits
    # ========================================================

    def encode_key(
        self,
        key: np.ndarray,
    ) -> np.ndarray:
        """
        Encode K information bits using a zero-frozen
        non-systematic polar code.

        Input:
            key: [B,K]

        Output:
            codeword: [B,N]
        """

        key = self._validate_bits(
            key,
            self.k,
            "key",
        )

        batch_size = key.shape[0]

        u = np.zeros(
            (batch_size, self.n),
            dtype=np.uint8,
        )

        u[:, self.info_pos] = key

        return polar_transform_bits(u)

    # ========================================================
    # 5. Main enrollment
    # ========================================================

    def enroll(
        self,
        batch_size: int = 1,
        *,
        rng: np.random.Generator | None = None,
        x: np.ndarray | None = None,
        scheme: str = "syndrome",
        key: np.ndarray | None = None,
    ) -> EnrollmentResult:
        """
        Enroll SRAM-PUF responses.

        Parameters
        ----------
        batch_size:
            Number of PUF responses to generate when
            x is not supplied.

        rng:
            NumPy random number generator.

        x:
            Optional fixed PUF responses [B,N].

        scheme:
            "syndrome", "transform", or "code_offset".

        key:
            Optional independent key [B,K].
            Accepted only for code_offset enrollment.

        Returns
        -------
        EnrollmentResult
        """

        rng = (
            np.random.default_rng()
            if rng is None
            else rng
        )

        scheme = str(
            scheme
        ).strip().lower()

        # "transform" is an alias for the existing scheme.
        if scheme == "transform":
            scheme = "syndrome"

        if scheme not in (
            "syndrome",
            "code_offset",
        ):
            raise ValueError(
                "scheme must be 'syndrome', "
                "'transform', or 'code_offset'."
            )

        # ----------------------------------------------------
        # Generate or validate physical PUF responses
        # ----------------------------------------------------

        if x is None:

            if batch_size < 1:
                raise ValueError(
                    "batch_size must be positive."
                )

            x = sample_puf_bits(
                self.n,
                batch_size,
                self.prob_one,
                rng,
            )

            x = self._validate_bits(
                x,
                self.n,
                "x",
            )

        else:

            x = self._validate_bits(
                x,
                self.n,
                "x",
            )

        B = x.shape[0]

        # Always retain the physical response's
        # polar-domain representation.
        c = polar_transform_bits(x)

        # ====================================================
        # Scheme A: syndrome / transform
        # ====================================================

        if scheme == "syndrome":

            if key is not None:
                raise ValueError(
                    "A supplied key is supported only "
                    "for code_offset enrollment."
                )

            # Key derived from the physical PUF.
            secret = c[:, self.info_pos].copy()

            # Public helper information.
            helper = c[:, self.frozen_pos].copy()

            # Embed helper bits at frozen positions.
            frozen_full = np.zeros(
                (B, self.n),
                dtype=np.uint8,
            )

            frozen_full[:, self.frozen_pos] = helper

            # Full-length reconstruction offset:
            # Z = W_full G_N
            offset = polar_transform_bits(
                frozen_full
            )

            # Zero-frozen codeword corresponding
            # to the enrolled source key.
            codeword = self.encode_key(
                secret
            )

            # Verify the fundamental algebraic identity.
            if not np.array_equal(
                x ^ codeword,
                offset,
            ):
                raise RuntimeError(
                    "Syndrome enrollment offset "
                    "consistency check failed."
                )

        # ====================================================
        # Scheme B: conventional code offset
        # ====================================================

        else:

            # ------------------------------------------------
            # Generate an independent secret key
            # ------------------------------------------------

            if key is None:

                secret = rng.integers(
                    low=0,
                    high=2,
                    size=(B, self.k),
                    dtype=np.uint8,
                )

            else:

                secret = self._validate_bits(
                    key,
                    self.k,
                    "key",
                )

                if secret.shape[0] != B:
                    raise ValueError(
                        "key and x must have the same "
                        "batch size."
                    )

            # ------------------------------------------------
            # Encode with zero frozen bits
            # ------------------------------------------------

            codeword = self.encode_key(
                secret
            )

            # ------------------------------------------------
            # Conventional public code offset
            # ------------------------------------------------

            offset = np.bitwise_xor(
                x,
                codeword,
            )

            # The full N-bit offset is public helper data.
            helper = offset.copy()

            # The offset is not representable as the
            # embedded frozen helper of this scheme.
            frozen_full = None

        # ====================================================
        # 6. Generate verification tag
        # ====================================================

        tag = self.checker.tag(
            secret
        )

        # ====================================================
        # 7. Return enrollment record
        # ====================================================

        return EnrollmentResult(
            x=x,
            c=c,
            helper=helper,
            key=secret,
            tag=tag,
            frozen_full=frozen_full,
            scheme=scheme,
            codeword=codeword,
            offset=offset,
        )

    # ========================================================
    # 8. Paired code-offset representation
    # ========================================================

    def code_offset(
        self,
        record: EnrollmentResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a code-offset representation using
        the SAME enrolled secret as record.key.

        This preserves the behavior of the original
        code_offset() method.

        Returns
        -------
        codeword:
            [B,N] Zero-frozen polar codeword.

        helper_offset:
            [B,N] X XOR codeword.

        For a syndrome enrollment record, this offset
        equals the polar transform of frozen_full.
        """

        if record.x.shape[1] != self.n:
            raise ValueError(
                "Record code length does not match."
            )

        codeword = self.encode_key(
            record.key
        )

        helper_offset = np.bitwise_xor(
            record.x,
            codeword,
        )

        if record.scheme == "syndrome":

            expected_offset = polar_transform_bits(
                record.frozen_full
            )

            if not np.array_equal(
                helper_offset,
                expected_offset,
            ):
                raise RuntimeError(
                    "Paired code-offset equivalence failed."
                )

        return codeword, helper_offset


# ============================================================
# Backward-compatible aliases
# ============================================================

PUFEnrollment = SRAMPUFEnrollment
