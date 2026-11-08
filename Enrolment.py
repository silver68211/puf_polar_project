"""SRAM-PUF enrollment for Chen et al. (2017) polar syndrome construction.

No assumptions about a uniformly random key are made for prob_one != 0.5.
For biased sources, the 2019 Chen-Willems secrecy-aware polar construction
requires additional index sets and analysis; this class is a baseline only.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from utils import sample_puf_bits, polar_transform_bits
from keycheck import KeyChecker


@dataclass
class EnrollmentResult:
    x: np.ndarray  # [B,N] raw SRAM PUF bits
    c: np.ndarray  # [B,N] polar-domain bits = X G_N
    helper: np.ndarray  # [B,N-K] publicly released syndrome bits
    key: np.ndarray  # [B,K] extracted source key
    tag: np.ndarray  # [B,T] public hash digest bytes or CRC parity bits
    frozen_full: np.ndarray  # [B,N] helper bits at frozen indices


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

        if n < 2 or n & (n - 1):
            raise ValueError("n must be a power of two.")

        positions = np.asarray(frozen_pos, dtype=np.int32)

        if positions.ndim != 1 or len(np.unique(positions)) != len(positions):
            raise ValueError("frozen_pos must contain unique indices.")

        if np.any(positions < 0) or np.any(positions >= n):
            raise ValueError("Frozen index out of bounds.")

        self.n = n
        self.frozen_pos = np.sort(positions)
        self.info_pos = np.setdiff1d(np.arange(n), self.frozen_pos)
        self.biased = bool(biased)
        self.prob_one = float(prob_one) if biased else 0.5

        if not 0 < self.prob_one < 1:
            raise ValueError("prob_one must be in (0,1)")
        self.checker = KeyChecker("hash") if checker is None else checker

    def enroll(
        self,
        batch_size: int = 1,
        *,
        rng: np.random.Generator | None = None,
        x: np.ndarray | None = None,
    ) -> EnrollmentResult:
        rng = np.random.default_rng() if rng is None else rng

        if x is None:
            x = sample_puf_bits(self.n, 1, self.prob_one, rng)

        else:
            x = np.asarray(x)
            if x.ndim == 1:
                x = x[None, :]
            if x.ndim != 2 or x.shape[1] != self.n or not np.all((x == 0) | (x == 1)):
                raise ValueError(f"x must be [B,{self.n}] binary")
            x = x.astype(np.uint8)

        c = polar_transform_bits(x)

        helper = c[:, self.frozen_pos]
        key = c[:, self.info_pos]

        frozen_full = np.zeros_like(c)
        frozen_full[:, self.frozen_pos] = helper

        return EnrollmentResult(
            x=x,
            c=c,
            helper=helper,
            key=key,
            tag=self.checker.tag(key),
            frozen_full=frozen_full,
        )

    def code_offset(self, record: EnrollmentResult):
        """Paired code-offset construction with SAME source-derived key.

        U_F=0, U_info=S; public W_offset=X xor (U G).
        This permits per-trial comparison of the two schemes (unbiased case).
        """
        polar_u = np.zeros_like(record.c)
        polar_u[:, self.info_pos] = record.key
        codeword = polar_transform_bits(polar_u)
        helper_offset = np.bitwise_xor(record.x, codeword)
        return codeword, helper_offset


# Short aliases
PUFEnrollment = SRAMPUFEnrollment
