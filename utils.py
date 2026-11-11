

from __future__ import annotations

import os
import textwrap
import warnings

from functools import lru_cache
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ============================================================
# 1. Matplotlib settings
# ============================================================

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "text.usetex": False,
    }
)


# ============================================================
# 2. 5G polar-code reliability ranking
# ============================================================

@lru_cache(maxsize=4)
def _load_5g_channel_order(path: str) -> np.ndarray:
    """Load the 5G reliability table from worst to best."""

    table = np.genfromtxt(path, delimiter=";")

    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("Invalid 5G reliability CSV format.")

    table = table[:, :2]

    if not np.all(np.isfinite(table)):
        raise ValueError("5G table contains invalid values.")

    if not np.all(table == np.floor(table)):
        raise ValueError("5G table must contain integer values.")

    table = table.astype(np.int32)

    # Column 0: reliability rank.
    # Column 1: polar channel index.
    ranks = table[:, 0]
    channels = table[:, 1]

    if (
        len(channels) != 1024
        or np.unique(ranks).size != 1024
        or not np.array_equal(
            np.sort(channels),
            np.arange(1024),
        )
    ):
        raise ValueError("Invalid 5G reliability table.")

    order = np.argsort(ranks, kind="stable")

    return channels[order]


def generate_5g_ranking(
    k: int,
    n: int,
    sort: bool = True,
    csv_path: str | Path = "codes/polar_5G.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return frozen and information positions using the 5G ranking.

    Parameters
    ----------
    k : int
        Number of information bits.
    n : int
        Code length, power of two in [32, 1024].
    sort : bool
        Return positions sorted in ascending index order.
    csv_path : str or Path
        Path to the 5G reliability table.

    Returns
    -------
    frozen_pos : np.ndarray
        Frozen positions, shape (n-k,).
    info_pos : np.ndarray
        Information positions, shape (k,).
    """

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")

    if not isinstance(sort, bool):
        raise TypeError("sort must be bool.")

    if not 32 <= n <= 1024 or n & (n - 1):
        raise ValueError(
            "n must be a power of two in [32, 1024]."
        )

    if not 0 <= k <= n:
        raise ValueError("Require 0 <= k <= n.")

    path = Path(csv_path)

    # Resolve relative paths from the project directory.
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.is_file():
        raise FileNotFoundError(
            f"5G ranking CSV not found: {path}"
        )

    reliability = _load_5g_channel_order(str(path.resolve()))

    # Keep the channels belonging to code length n.
    reliability = reliability[reliability < n]

    # Least reliable channels are frozen.
    frozen_pos = reliability[: n - k].copy()
    info_pos = reliability[n - k :].copy()

    if sort:
        frozen_pos = np.sort(frozen_pos)
        info_pos = np.sort(info_pos)

    return frozen_pos, info_pos


# ============================================================
# 3. Plot BLER / FER curves
# ============================================================

def plot_graph(
    fers: Dict[str, np.ndarray],
    title: str,
    xlabel: str = r"$E_b/N_0$ (dB)",
    ylabel: str = "BLER",
    loc: str = "lower left",
    marker: Optional[Sequence[str]] = None,
    color: Optional[Sequence[str]] = None,
    linestyle: Optional[Sequence[str]] = None,
    markevery: int = 1,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
    figsize: Tuple[float, float] = (6, 8),
    save: bool = False,
):
    """Plot multiple BLER curves on a logarithmic scale.

    Each input array must have shape (num_points, 2),
    where columns contain the x coordinate and BLER.
    """

    if color is None:
        color = [
            "black",
            "blue",
            "green",
            "red",
            "purple",
            "orange",
        ]

    if linestyle is None:
        linestyle = ["-", "--", "-.", ":"]

    if marker is None:
        marker = ["o", "s", "^", "D", "v", "x"]

    if not len(color) or not len(linestyle) or not len(marker):
        raise ValueError("Plot styles must not be empty.")

    fig, ax = plt.subplots(figsize=figsize)

    # Cycle styles to ensure every curve is plotted.
    for i, (label, arr) in enumerate(fers.items()):

        arr = np.asarray(arr)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Data for {label!r} must have shape (M, 2)."
            )

        ax.semilogy(
            arr[:, 0],
            arr[:, 1],
            marker=marker[i % len(marker)],
            color=color[i % len(color)],
            linestyle=linestyle[i % len(linestyle)],
            markevery=markevery,
            label=label,
            markerfacecolor="white",
            linewidth=1.8,
            markersize=9,
            markeredgewidth=2,
        )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        direction="in",
        length=6,
        width=1,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=1.5,
        width=0.5,
    )

    ax.grid(
        which="major",
        linestyle="-.",
        linewidth=0.8,
        color="gray",
        alpha=0.8,
    )

    ax.minorticks_on()

    ax.grid(
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.4,
        color="gray",
        alpha=0.2,
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(loc=loc, fontsize=14)

    fig.tight_layout()

    if save:
        os.makedirs("figs", exist_ok=True)

        fig.savefig(
            f"figs/{title}.pdf",
            dpi=300,
            bbox_inches="tight",
        )


# ============================================================
# 4. Plot polar-code construction bitmasks
# ============================================================

def plot_construction_bitmask_image(
    constructions: Dict[str, np.ndarray],
    dec: str = "SC",
    N: int = 64,
    height: int = 16,
    out_dir: str = "figs",
):
    """Plot binary masks representing polar-code constructions."""

    if N < 1 or height < 1 or N % height != 0:
        raise ValueError(
            "N and height must be positive, and N must be "
            "divisible by height."
        )

    if not constructions:
        raise ValueError("No constructions supplied.")

    num_constructions = len(constructions)
    width = N // height

    base_colors = [
        "blue",
        "red",
        "purple",
        "magenta",
        "orange",
        "green",
    ]

    cmaps = [
        mcolors.LinearSegmentedColormap.from_list(
            f"white_{c}",
            ["white", c],
            N=2,
        )
        for c in base_colors
    ]

    pixel_size = 0.11

    fig_height = (
        pixel_size * height * num_constructions + 1.0
    )

    fig_width = pixel_size * width + 1.0

    fig, axes = plt.subplots(
        num_constructions,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]
    legend_handles = []

    for i, (ax, (label, mask)) in enumerate(
        zip(axes, constructions.items())
    ):

        mask = np.asarray(mask).reshape(-1)

        if mask.size != N:
            raise ValueError(
                f"Mask for {label!r} must have length {N}."
            )

        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Construction masks must be binary.")

        grid = mask.reshape(width, height).T

        cmap = cmaps[i % len(cmaps)]

        ax.imshow(
            grid,
            cmap=cmap,
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
            origin="lower",
        )

        ax.set_xticks(
            np.arange(-0.5, width, 1),
            minor=True,
        )

        ax.set_yticks(
            np.arange(-0.5, height, 1),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="gray",
            linestyle="-.",
            linewidth=0.2,
        )

        ax.tick_params(which="both", length=0)

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)

        ax.set_xticks([])

        # Left labels.
        ax.set_yticks([0, height - 1])

        ax.set_yticklabels(
            [height, 1],
            fontsize=14,
        )

        # Right labels.
        sec = ax.secondary_yaxis("right")

        sec.set_yticks([0, height - 1])

        sec.set_yticklabels(
            [str(N), str(N - height)],
            fontsize=14,
        )

        sec.tick_params(length=0)

        legend_handles.append(
            mpatches.Patch(
                facecolor=cmap(1.0),
                edgecolor="black",
                label=textwrap.fill(label, width=18),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=1.0,
        handleheight=1.0,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"bitmask_{N}_{dec}.pdf",
    )

    fig.savefig(
        out_path,
        bbox_inches="tight",
    )

    plt.show()

    return out_path


# ============================================================
# 5. Polar transform matrix
# ============================================================

def generate_polar_transform_mat(
    n_lift: int,
) -> np.ndarray:
    """Generate G_N = F^(tensor n_lift) over GF(2).

    F = [[1, 0], [1, 1]].
    No bit reversal is applied.
    """

    if not isinstance(n_lift, (int, np.integer)):
        raise TypeError("n_lift must be an integer.")

    n_lift = int(n_lift)

    if n_lift < 0:
        raise ValueError("n_lift must be nonnegative.")

    if n_lift >= 12:
        warnings.warn(
            "n_lift >= 12 produces a large dense matrix.",
            stacklevel=2,
        )

    F = np.array(
        [[1, 0], [1, 1]],
        dtype=np.int8,
    )

    gm = np.array([[1]], dtype=np.int8)

    for _ in range(n_lift):
        gm = np.kron(gm, F)

    return gm


# ============================================================
# 6. Fast polar transform
# ============================================================

def polar_transform_bits(bits: np.ndarray) -> np.ndarray:
    """Compute x = u G_N over GF(2).

    G_N = F^(tensor log2(N)).
    No bit reversal is applied.

    Supports:
        [N]
        [batch_size, N]

    The transform is self-inverse.
    """

    u = np.asarray(bits)

    if u.ndim not in (1, 2):
        raise ValueError(
            "Expected shape [N] or [batch_size, N]."
        )

    N = u.shape[-1]

    if N < 2 or N & (N - 1):
        raise ValueError(
            "Polar transform length must be a power of two >= 2."
        )

    if not np.all((u == 0) | (u == 1)):
        raise ValueError("Input must contain binary bits.")

    x = u.astype(np.uint8, copy=True)

    half = 1

    while half < N:

        blocks = x.reshape(
            *x.shape[:-1],
            -1,
            2 * half,
        )

        blocks[..., :half] ^= blocks[..., half:]

        half *= 2

    return x


# ============================================================
# 7. Reed-Muller construction
# ============================================================

def generate_rm_code(r: int, m: int):
    """Generate frozen and information positions for RM(r,m).

    Frozen indices have Hamming weight < m-r.

    Returns:
        frozen_pos, info_pos, n, k, d_min
    """

    if not isinstance(r, (int, np.integer)):
        raise TypeError("r must be an integer.")

    if not isinstance(m, (int, np.integer)):
        raise TypeError("m must be an integer.")

    r = int(r)
    m = int(m)

    if r < 0 or m < 0:
        raise ValueError("r and m must be nonnegative.")

    if r > m:
        raise ValueError("Require r <= m.")

    n = 1 << m
    d_min = 1 << (m - r)

    # Exact integer dimension.
    k = sum(comb(m, i) for i in range(r + 1))

    # Compute Hamming weights of the binary indices.
    dtype = (
        np.uint16
        if m <= 16
        else np.uint32
        if m <= 32
        else np.uint64
    )

    indices = np.arange(n, dtype=dtype)

    byte_view = indices.view(np.uint8).reshape(n, -1)

    weights = np.unpackbits(
        byte_view,
        axis=1,
    ).sum(axis=1)

    frozen_pos = np.flatnonzero(
        weights < (m - r)
    )

    info_pos = np.flatnonzero(
        weights >= (m - r)
    )

    if info_pos.size != k:
        raise RuntimeError(
            f"Inconsistent RM dimension: "
            f"{info_pos.size} != {k}."
        )

    return frozen_pos, info_pos, n, k, d_min


# ============================================================
# 8. Dense polar generator and parity-check matrices
# ============================================================

def generate_dense_polar(
    frozen_pos: np.ndarray,
    n: int,
    verbose: bool = True,
):
    """Generate the dense parity-check and generator matrices.

    Returns:
        pcm : [N-K, N]
        gm_true : [K, N]

    Verifies H G^T = 0 over GF(2).
    """

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")

    n = int(n)

    if n < 2 or n & (n - 1):
        raise ValueError(
            "n must be a power of two >= 2."
        )

    frozen_pos = np.asarray(frozen_pos)

    if frozen_pos.ndim != 1:
        raise ValueError(
            "frozen_pos must be one-dimensional."
        )

    if frozen_pos.size == 0:
        frozen_pos = frozen_pos.astype(np.int64)

    if not np.issubdtype(
        frozen_pos.dtype,
        np.integer,
    ):
        raise TypeError(
            "frozen_pos must contain integers."
        )

    if np.any(
        (frozen_pos < 0) | (frozen_pos >= n)
    ):
        raise ValueError("Invalid frozen positions.")

    if np.unique(frozen_pos).size != frozen_pos.size:
        raise ValueError("Duplicate frozen positions.")

    k = n - frozen_pos.size

    info_pos = np.setdiff1d(
        np.arange(n),
        frozen_pos,
    )

    if info_pos.size != k:
        raise RuntimeError(
            "Invalid information positions."
        )

    gm_mat = generate_polar_transform_mat(
        n.bit_length() - 1
    ).astype(np.int64)

    # Generator rows at information positions.
    gm_true = gm_mat[info_pos, :]

    # Parity-check rows associated with frozen positions.
    pcm = gm_mat[:, frozen_pos].T

    # Verify orthogonality over GF(2).
    syndrome = (pcm @ gm_true.T) & 1

    if np.any(syndrome):
        raise RuntimeError(
            "Invalid generator/parity-check matrix pair."
        )

    if verbose:

        print(
            "Shape of generator matrix:",
            gm_true.shape,
        )

        print(
            "Shape of parity-check matrix:",
            pcm.shape,
        )

        plt.spy(pcm)

    return pcm, gm_true


# ============================================================
# 9. Bit / integer conversion
# ============================================================

def bin2int(
    arr: List[int] | np.ndarray,
) -> Optional[int]:
    """Convert MSB-first binary bits to an integer."""

    if len(arr) == 0:
        return None

    bits = np.asarray(arr)

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Input must contain binary bits.")

    return int(
        "".join(str(int(bit)) for bit in bits),
        2,
    )


def bin2int_tf(arr: tf.Tensor) -> tf.Tensor:
    """Convert MSB-first binary tensors to int64."""

    arr = tf.cast(arr, tf.int64)

    length = tf.shape(arr)[-1]

    tf.debugging.assert_less_equal(
        length,
        63,
        message="Signed int64 conversion supports up to 63 bits.",
    )

    shifts = tf.range(
        tf.cast(length, tf.int64) - 1,
        -1,
        -1,
        dtype=tf.int64,
    )

    return tf.reduce_sum(
        tf.bitwise.left_shift(arr, shifts),
        axis=-1,
    )


def int2bin(num: int, len_: int) -> List[int]:
    """Convert an integer to a fixed-length MSB-first bit list."""

    if num < 0:
        raise ValueError("num must be nonnegative.")

    if len_ < 0:
        raise ValueError("len_ must be nonnegative.")

    if len_ == 0:
        return []

    return [
        int(bit)
        for bit in format(num, f"0{len_}b")[-len_:]
    ]


def int2bin_tf(
    ints: tf.Tensor,
    len_: int,
) -> tf.Tensor:
    """Convert integers to MSB-first bit tensors."""

    if not 0 <= len_ <= 63:
        raise ValueError(
            "len_ must be in [0, 63] for signed int64."
        )

    ints = tf.cast(ints, tf.int64)

    shifts = tf.range(
        len_ - 1,
        -1,
        -1,
        dtype=tf.int64,
    )

    shifted = tf.bitwise.right_shift(
        tf.expand_dims(ints, -1),
        shifts,
    )

    return tf.bitwise.bitwise_and(
        shifted,
        tf.constant(1, dtype=tf.int64),
    )


# ============================================================
# 10. Alist utilities
# ============================================================

def alist2mat(alist, verbose=True):
    """Convert an alist representation to a parity-check matrix."""

    if len(alist) <= 4:
        raise ValueError("Invalid alist format.")

    n, m = alist[0][:2]

    v_max, c_max = alist[1][:2]

    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    if len(vn_profile) != n or len(cn_profile) != m:
        raise ValueError("Invalid alist degree profiles.")

    if np.sum(vn_profile) != np.sum(cn_profile):
        raise ValueError("Inconsistent alist edge counts.")

    if (
        np.max(vn_profile) != v_max
        or np.max(cn_profile) != c_max
    ):
        raise ValueError("Invalid maximum node degrees.")

    vn_only = len(alist) == n + 4

    if not vn_only and len(alist) != n + m + 4:
        raise ValueError("Invalid number of alist rows.")

    if vn_only and verbose:
        print(
            "Note: alist does not contain CN perspective."
        )

    pcm = np.zeros((m, n), dtype=np.int8)

    num_edges = 0

    # Variable-node connections.
    for idx_v in range(n):

        degree = vn_profile[idx_v]

        for idx_i in range(degree):

            idx_c = alist[4 + idx_v][idx_i] - 1

            if not 0 <= idx_c < m:
                raise ValueError(
                    "Invalid check-node index."
                )

            pcm[idx_c, idx_v] = 1

            num_edges += 1

    # Validate check-node connections, if available.
    if not vn_only:

        for idx_c in range(m):

            degree = cn_profile[idx_c]

            for idx_i in range(degree):

                idx_v = (
                    alist[4 + n + idx_c][idx_i] - 1
                )

                if not 0 <= idx_v < n:
                    raise ValueError(
                        "Invalid variable-node index."
                    )

                if pcm[idx_c, idx_v] != 1:
                    raise ValueError(
                        "Inconsistent VN/CN connections."
                    )

    if not np.array_equal(
        pcm.sum(axis=0),
        vn_profile,
    ):
        raise ValueError(
            "Variable-node degree mismatch."
        )

    if not np.array_equal(
        pcm.sum(axis=1),
        cn_profile,
    ):
        raise ValueError(
            "Check-node degree mismatch."
        )

    if verbose:

        print(
            "n:", n,
            "m:", m,
            "k:", k,
            "rate:", coderate,
        )

        print(
            "edges:", num_edges,
            "v_max:", v_max,
            "c_max:", c_max,
        )

        plt.spy(pcm)

    return pcm.astype(int), k, n, coderate


def load_alist(path: str | Path) -> list:
    """Load an alist file."""

    alist = []

    with open(path, "r", encoding="utf-8") as reader:

        for line in reader:

            row = [int(word) for word in line.split()]

            if row:
                alist.append(row)

    return alist


# ============================================================
# 11. TensorFlow modulo-2
# ============================================================

def int_mod_2(x: tf.Tensor) -> tf.Tensor:
    """Compute modulo 2 using a bitwise AND."""

    if not x.dtype.is_integer:
        x = tf.cast(x, tf.int64)

    return tf.bitwise.bitwise_and(
        x,
        tf.constant(1, dtype=x.dtype),
    )


# ============================================================
# 12. Bhattacharyya parameter
# ============================================================

@lru_cache(maxsize=None)
def bhattacharyya_parameter(
    i: int,
    N: int,
    alpha: float,
) -> float:
    """Compute a synthetic-channel Bhattacharyya parameter.

    i is the 1-based synthetic-channel index.

    Exact for BEC; upper-bound recursion for BSC/AWGN
    when alpha is the corresponding initial parameter.
    """

    if N < 1 or N & (N - 1):
        raise ValueError(
            "N must be a positive power of two."
        )

    if not 1 <= i <= N:
        raise ValueError(
            "Require 1 <= i <= N."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be in [0, 1]."
        )

    if N == 1:
        return float(alpha)

    half = N // 2

    if i % 2 == 0:

        z = bhattacharyya_parameter(
            i // 2,
            half,
            alpha,
        )

        return z * z

    z = bhattacharyya_parameter(
        (i + 1) // 2,
        half,
        alpha,
    )

    return 2.0 * z - z * z


# ============================================================
# 13. AWGN noise standard deviation
# ============================================================

def snr_db_to_sigma(snr_db: float) -> float:
    """Convert Es/N0 in dB to the AWGN noise standard deviation.

    Assumes unit-energy BPSK:

        sigma^2 = 1 / (2 * Es/N0)
    """

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    snr_linear = 10.0 ** (snr_db / 10.0)

    return float(
        np.sqrt(1.0 / (2.0 * snr_linear))
    )


# ============================================================
# 14. SRAM-PUF response generation
# ============================================================

def sample_puf_bits(
    n: int,
    batch_size: int = 1,
    prob_one: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample independent SRAM cells X_i ~ Bernoulli(prob_one)."""

    if n <= 0 or batch_size <= 0:
        raise ValueError(
            "n and batch_size must be positive."
        )

    if not 0.0 < prob_one < 1.0:
        raise ValueError(
            "prob_one must be in (0, 1)."
        )

    rng = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    return (
        rng.random((batch_size, n)) < prob_one
    ).astype(np.uint8)


# ============================================================
# 15. Communication channel simulation
# ============================================================

def sample_channel(
    bits: np.ndarray,
    channel: str = "BSC",
    *,
    p: float = 0.15,
    snr_db: float = 0.0,
    erasure_probability: float = 0.15,
    prob_one: float = 0.5,
    llr_clip: float = 80.0,
    rng: Optional[np.random.Generator] = None,
):
    """Generate channel observations and physical-bit LLRs.

    Supported channels:
        BSC
        AWGN
        BEC

    LLR convention:
        L = log[P(X=0|Y) / P(X=1|Y)]

        Positive LLR favors bit 0.
        Negative LLR favors bit 1.

    prob_one is the physical PUF source bias q=P(X=1).

    Returns:
        y : channel observations
        llr : float32 LLR array

    Note:
        The source prior does not account for conditioning
        on public enrollment helper data.
    """

    rng = (
        np.random.default_rng()
        if rng is None
        else rng
    )

    channel = channel.strip().upper()

    x = np.asarray(bits)

    if (
        x.ndim not in (1, 2)
        or not np.all((x == 0) | (x == 1))
    ):
        raise ValueError(
            "bits must be a binary array [N] or [B, N]."
        )

    if not 0.0 < prob_one < 1.0:
        raise ValueError(
            "prob_one must be in (0, 1)."
        )

    if not np.isfinite(llr_clip) or llr_clip <= 0:
        raise ValueError(
            "llr_clip must be finite and positive."
        )

    # Physical PUF source prior:
    #
    # log[P(X=0) / P(X=1)]
    #
    # Zero for unbiased SRAM-PUFs.
    prior = np.log1p(-prob_one) - np.log(prob_one)

    # --------------------------------------------------------
    # Binary Symmetric Channel
    # --------------------------------------------------------

    if channel == "BSC":

        if not 0.0 <= p < 0.5:
            raise ValueError(
                "BSC crossover p must be in [0, 0.5)."
            )

        # Independent bit errors.
        noise = (
            rng.random(x.shape) < p
        ).astype(np.uint8)

        # Y = X XOR E.
        y = x.astype(np.uint8) ^ noise

        # Channel LLR magnitude.
        #
        # At p=0, use a finite value that dominates
        # the source prior.
        magnitude = (
            llr_clip + abs(prior)
            if p == 0.0
            else np.log1p(-p) - np.log(p)
        )

        llr = (
            (1.0 - 2.0 * y.astype(np.float64))
            * magnitude
            + prior
        )

    # --------------------------------------------------------
    # Additive White Gaussian Noise Channel
    # --------------------------------------------------------

    elif channel == "AWGN":

        sigma = snr_db_to_sigma(snr_db)

        # BPSK mapping:
        # 0 -> +1
        # 1 -> -1
        symbols = (
            1.0 - 2.0 * x.astype(np.float64)
        )

        noise = rng.normal(
            0.0,
            sigma,
            x.shape,
        )

        y = symbols + noise

        llr = (
            2.0 * y / sigma**2
            + prior
        )

    # --------------------------------------------------------
    # Binary Erasure Channel
    # --------------------------------------------------------

    elif channel == "BEC":

        eps = erasure_probability

        if not 0.0 <= eps <= 1.0:
            raise ValueError(
                "erasure_probability must be in [0, 1]."
            )

        erased = rng.random(x.shape) < eps

        # Erased observations are represented by -1.
        y = x.astype(np.int8).copy()

        y[erased] = -1

        # Unerased bits are known exactly.
        llr = (
            1.0 - 2.0 * x.astype(np.float64)
        ) * llr_clip

        # Erased bits contain only the source prior.
        llr[erased] = prior

    else:

        raise ValueError(
            "channel must be BSC, AWGN, or BEC."
        )

    llr = np.clip(
        llr,
        -llr_clip,
        llr_clip,
    ).astype(np.float32)

    return y, llr


# ============================================================
# 16. Backward-compatible channel alias
# ============================================================

bsc_awgn_bec = sample_channel
