
"""Polar-code construction methods: GA, RM, RM-Polar, and Bhattacharyya."""

from collections import Counter

import numpy as np


# ============================================================
# 1. Gaussian Approximation (GA)
# ============================================================

class GA:
    """Gaussian Approximation construction for polar codes."""

    def __init__(
        self,
        snr_db: float,
        n_log2: int = 7,
        rate: float = 0.5,
    ):
        if n_log2 < 1:
            raise ValueError("n_log2 must be >= 1.")

        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1].")

        self.n_log2 = int(n_log2)
        self.N = 1 << self.n_log2
        self.K = int(round(rate * self.N))
        self.rate = float(rate)

        if self.K < 1:
            raise ValueError("K must be positive.")

        # Initial mean LLR for the AWGN approximation.
        self.design_llr = (
            4.0 * self.rate * 10.0 ** (snr_db / 10.0)
        )

        initial = np.full(
            self.N,
            self.design_llr,
            dtype=np.float64,
        )

        self.sequence = self.gaussian_approx(initial)

        # Smaller mean LLR means lower reliability.
        order = np.argsort(self.sequence, kind="stable")

        self.frozen_pos = np.sort(order[: self.N - self.K])
        self.info_pos = np.sort(order[self.N - self.K :])

    @staticmethod
    def phi_new(x: np.ndarray) -> np.ndarray:
        """Approximation of the GA phi function."""

        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)

        mask = x < 10.0

        y[mask] = np.exp(
            -0.4527 * x[mask] ** 0.86 + 0.0218
        )

        xm = np.maximum(x[~mask], 1e-12)

        y[~mask] = (
            np.sqrt(np.pi / xm)
            * (1.0 - 10.0 / (7.0 * xm))
            * np.exp(-xm / 4.0)
        )

        return np.clip(y, 1e-15, 1.0)

    @classmethod
    def phi_inv(
        cls,
        y: np.ndarray,
        lo: float = 0.0,
        hi: float = 10_000.0,
        tol: float = 1e-2,
    ) -> np.ndarray:
        """Numerical inverse of the phi approximation."""

        y = np.clip(
            np.asarray(y, dtype=np.float64),
            1e-15,
            1.0,
        )

        a = np.full_like(y, lo)
        b = np.full_like(y, hi)

        while np.max(b - a) >= tol:
            c = 0.5 * (a + b)

            fa = cls.phi_new(a) - y
            fc = cls.phi_new(c) - y

            left = (fa * fc) < 0

            b = np.where(left, c, b)
            a = np.where(left, a, c)

        return 0.5 * (a + b)

    def gaussian_approx(
        self,
        init_mean: np.ndarray,
    ) -> np.ndarray:
        """Compute synthetic-channel mean LLRs."""

        cur = np.asarray(
            init_mean,
            dtype=np.float64,
        ).copy()

        if cur.shape != (self.N,):
            raise ValueError(
                f"init_mean must have shape ({self.N},)."
            )

        for stage in range(1, self.n_log2 + 1):
            half = 1 << (stage - 1)
            size = 2 * half

            prev = cur[:half]

            phi = self.phi_new(prev)

            # Less reliable synthetic channel.
            upper = self.phi_inv(
                1.0 - (1.0 - phi) ** 2
            )

            # More reliable synthetic channel.
            lower = 2.0 * prev

            nxt = np.empty(size, dtype=np.float64)
            nxt[0::2] = upper
            nxt[1::2] = lower

            cur[:size] = nxt

        return cur


# ============================================================
# 2. Reed-Muller (RM) construction
# ============================================================

def bin_rep(n: int) -> np.ndarray:
    """Binary representations of 0,...,2^n-1."""

    if n < 1:
        raise ValueError("n must be >= 1.")

    indices = np.arange(
        1 << n,
        dtype=np.uint64,
    )[:, None]

    shifts = np.arange(n, dtype=np.uint64)

    return ((indices >> shifts) & 1).astype(np.uint8)


def RM_Design_Rule(N: int, K: int) -> np.ndarray:
    """Select K polar generator rows with largest Hamming weights."""

    if N < 2 or (N & (N - 1)):
        raise ValueError("N must be a power of two.")

    if not 0 < K <= N:
        raise ValueError("Require 0 < K <= N.")

    n = N.bit_length() - 1

    # Generator-row weights are 2^wt(i).
    # Ranking by wt(i) gives the same ordering.
    weights = bin_rep(n).sum(axis=1)

    order = np.argsort(-weights.astype(np.int32), kind="stable")

    return np.sort(order[:K].astype(np.int32))


# ============================================================
# 3. RM-Polar construction
# ============================================================

def RM_Polar_Design_Rule(
    N: int,
    K: int,
    crc: int = 0,
    ebno_db: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """RM filtering followed by GA-based reliability selection."""

    if N < 2 or (N & (N - 1)):
        raise ValueError("N must be a power of two.")

    target = K + crc

    if not (0 < K <= N and 0 < target <= N):
        raise ValueError("Require 0 < K and 0 < K+crc <= N.")

    n = N.bit_length() - 1

    # GA reliability.
    ga = GA(
        snr_db=ebno_db,
        n_log2=n,
        rate=K / N,
    )

    # Z-like reliability metric.
    # Equivalent to exp(-1/(2*sigma)), sigma=2/mean.
    zw = np.exp(
        -np.maximum(ga.sequence, 0.0) / 4.0
    )

    # Hamming weights of polar indices.
    weights = bin_rep(n).sum(axis=1)

    counts = Counter(weights)

    # Determine the lowest permitted RM weight.
    cumulative = 0
    minimum_weight = 0

    for weight in sorted(counts, reverse=True):
        cumulative += counts[weight]

        minimum_weight = weight

        if cumulative >= target:
            break

    # Eligible RM positions.
    candidates = np.where(
        weights >= minimum_weight
    )[0]

    # Select the most reliable eligible positions.
    order = np.argsort(
        zw[candidates],
        kind="stable",
    )

    info = candidates[order[:target]]

    return np.sort(info.astype(np.int32)), zw


# ============================================================
# 4. Bhattacharyya construction
# ============================================================

def bhattacharyya_design(
    n: int,
    k: int,
    *,
    channel: str = "BSC",
    design_p: float = 0.15,
    design_snr_db: float = 0.0,
    design_erasure: float = 0.15,
):
    """
    Construct polar codes using Bhattacharyya recursion.

    BEC:
        Exact recursion.

    BSC / AWGN:
        Conservative upper-bound recursion.

    Returns:
        frozen : [N-K] frozen indices
        info   : [K] information indices
        z      : [N] Bhattacharyya estimates/bounds
    """

    if n < 2 or (n & (n - 1)):
        raise ValueError("n must be a power of two.")

    if not 0 < k < n:
        raise ValueError("Require 0 < k < n.")

    channel = channel.upper()

    # --------------------------------------------------------
    # Initial Bhattacharyya parameter
    # --------------------------------------------------------

    if channel == "BSC":
        if not 0.0 <= design_p <= 0.5:
            raise ValueError("design_p must be in [0, 0.5].")

        z0 = 2.0 * np.sqrt(
            design_p * (1.0 - design_p)
        )

    elif channel == "BEC":
        if not 0.0 <= design_erasure <= 1.0:
            raise ValueError(
                "design_erasure must be in [0, 1]."
            )

        z0 = float(design_erasure)

    elif channel == "AWGN":
        # Equiprobable BPSK, unit symbol energy.
        z0 = np.exp(
            -(10.0 ** (design_snr_db / 10.0))
        )

    else:
        raise ValueError(
            "channel must be BSC, BEC, or AWGN."
        )

    # --------------------------------------------------------
    # Polarization recursion
    # --------------------------------------------------------

    z = np.array([z0], dtype=np.float64)

    while z.size < n:
        nxt = np.empty(2 * z.size, dtype=np.float64)

        # Minus channel.
        nxt[0::2] = np.minimum(
            1.0,
            2.0 * z - z * z,
        )

        # Plus channel.
        nxt[1::2] = z * z

        z = nxt

    # --------------------------------------------------------
    # Select most reliable channels
    # --------------------------------------------------------

    order = np.argsort(z, kind="stable")

    info = np.sort(
        order[:k].astype(np.int32)
    )

    frozen = np.sort(
        order[k:].astype(np.int32)
    )

    return frozen, info, z


# ============================================================
# 5. Example
# ============================================================

if __name__ == "__main__":

    N = 1024
    K = 512

    frozen, info, z = bhattacharyya_design(
        N,
        K,
        channel="BSC",
        design_p=0.15,
    )

    print(f"N={N}, K={K}")
    print("Information positions:", info)
    print("Frozen positions:", frozen)
