
"""SRAM-PUF polar Monte Carlo simulation using Python/TensorFlow decoders.

Supports:
    Enrollment: syndrome, code_offset
    Decoders: SC, DSCF
    Construction: 5G, BSC, BHATTACHARYYA, OURS
    Verification: SHA-256, CRC16
    Optional polar-codeword scrambling
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import SimConfig
from constructions import bhattacharyya_design
from DSCFlip import PolarSCDecoder, PolarDSCFlipDecoder
from Enrolment import SRAMPUFEnrollment
from keycheck import KeyChecker
from utils import generate_5g_ranking, polar_transform_bits


# ============================================================
# 1. Polar-code construction
# ============================================================

def construct_polar_code(cfg, n):

    k = cfg.key_length
    construction = cfg.construction.strip().upper()

    if construction == "5G":

        frozen, info = generate_5g_ranking(k, n)

    elif construction in ("BSC", "BHATTACHARYYA"):

        frozen, info, _ = bhattacharyya_design(
            n,
            k,
            channel="BSC",
            design_p=cfg.design_p,
        )

    elif construction == "OURS":

        path = (
            Path(__file__).resolve().parent
            / "codes"
            / f"frozen_bits_N{n}_K{k}_our.csv"
        )

        frozen = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.int64,
        ).reshape(-1)

        info = np.setdiff1d(
            np.arange(n),
            frozen,
        )

    else:
        raise ValueError(
            f"Unsupported construction: {construction}"
        )

    frozen = np.sort(
        np.asarray(frozen, dtype=np.int64)
    )

    info = np.sort(
        np.asarray(info, dtype=np.int64)
    )

    # Verify that the frozen and information indices
    # form a complete partition of [0, N).

    if (
        frozen.shape != (n - k,)
        or info.shape != (k,)
        or not np.array_equal(
            np.sort(
                np.concatenate((frozen, info))
            ),
            np.arange(n),
        )
    ):
        raise ValueError(
            "Invalid frozen/information position partition."
        )

    return frozen, info


# ============================================================
# 2. Configuration and key verification
# ============================================================

def get_schemes(cfg):

    schemes = cfg.enrollment_schemes

    if isinstance(schemes, str):
        schemes = (schemes,)

    schemes = tuple(
        s.strip().lower()
        for s in schemes
    )

    return tuple(
        "syndrome" if s == "transform" else s
        for s in schemes
    )


def get_decoders(cfg):

    decoders = cfg.decoder

    if isinstance(decoders, str):
        decoders = (decoders,)

    decoders = tuple(
        d.strip().upper()
        for d in decoders
    )

    if any(
        d not in ("SC", "DSCF")
        for d in decoders
    ):
        raise ValueError(
            "Python simulation supports SC and DSCF only. "
            "Use sim_aff3ct.py for SCL."
        )

    return decoders


def build_checker(cfg):

    return KeyChecker(
        method=cfg.checker.lower(),
        hash_name=cfg.hash_name,
        hash_bits=cfg.hash_bits,
        crc_degree=cfg.crc_degree,
    )


class MaskAwareChecker:
    """Remove information scrambling before tag verification.

    Each verification row contains:

        [public_tag, information_mask]

    DSCF carries this row with the corresponding candidate
    during its retry attempts.
    """

    def __init__(self, checker, k):

        self.checker = checker
        self.k = k

    def verify(self, candidates, metadata):

        candidates = np.asarray(
            candidates,
            dtype=np.uint8,
        )

        metadata = np.asarray(
            metadata,
            dtype=np.uint8,
        )

        tags = metadata[:, :-self.k]
        masks = metadata[:, -self.k:]

        # Recover the original candidate secret.
        original_keys = candidates ^ masks

        return self.checker.verify(
            original_keys,
            tags,
        )


# ============================================================
# 3. Enroll one fixed SRAM-PUF device
# ============================================================

def enroll_devices(
    cfg,
    n,
    frozen,
    checker,
    schemes,
    device_rng,
    key_rng,
):

    q = cfg.puf_one_probability

    enrollment = SRAMPUFEnrollment(
        n=n,
        frozen_pos=frozen,
        biased=(q != 0.5),
        prob_one=q,
        checker=checker,
    )

    # Generate exactly ONE physical SRAM-PUF response.
    # The same response is used by every scheme.

    if cfg.all_zero_device:

        x = np.zeros(
            (1, n),
            dtype=np.uint8,
        )

    else:

        x = (
            device_rng.random((1, n)) < q
        ).astype(np.uint8)

    # Syndrome:
    #     Secret derived from the polar transform of X.
    #
    # Code offset:
    #     An independently generated secret.

    devices = {
        scheme: enrollment.enroll(
            x=x,
            rng=key_rng,
            scheme=scheme,
        )
        for scheme in schemes
    }

    return x, devices


# ============================================================
# 4. Python decoders
# ============================================================

def build_decoder(
    cfg,
    n,
    frozen,
    checker,
    decoder_name,
):

    if decoder_name == "SC":

        return PolarSCDecoder(
            frozen_pos=frozen,
            n=n,
            input_is_logit=False,
            tie_breaking="one",
        )

    if decoder_name == "DSCF":

        return PolarDSCFlipDecoder(
            frozen_pos=frozen,
            n=n,
            omga=1,

            checker=MaskAwareChecker(
                checker,
                cfg.key_length,
            ),

            alpha=cfg.alpha,
            num_trials=cfg.num_trials,

            use_flip_sc=True,
            input_is_logit=False,
            tie_breaking="one",
        )

    raise ValueError(
        f"Unsupported decoder: {decoder_name}"
    )


def decode_batch(
    decoder,
    decoder_name,
    llrs,
    expected_tag,
    info,
    masks,
):
    """Decode one batch of offset-corrected observations."""

    llrs = tf.convert_to_tensor(
        llrs,
        dtype=tf.float32,
    )

    if decoder_name == "SC":

        decoded, _ = decoder(llrs)

    else:

        tags = np.broadcast_to(
            expected_tag,
            (
                len(masks),
                expected_tag.shape[1],
            ),
        )

        # Carry both the public tag and the corresponding
        # scrambling mask through the DSCF retry process.

        metadata = np.concatenate(
            (tags, masks),
            axis=1,
        )

        # Direct eager call avoids symbolic tracing
        # of the Python hash verification function.

        decoded, _ = decoder.call(
            llrs,
            expected_tags=metadata,
        )

    # Extract information bits.
    estimated = decoded.numpy()[
        :, info
    ].astype(np.uint8)

    # Recover the original enrolled secret.
    return estimated ^ masks


# ============================================================
# 5. BSC observations and optional scrambling
# ============================================================

def bsc_llrs(
    x,
    p,
    q,
    batch,
    rng,
):
    """Generate BSC observations and physical-PUF LLRs."""

    errors = rng.random(
        (batch, x.shape[1])
    ) < p

    y = x ^ errors.astype(np.uint8)

    # Physical SRAM-PUF prior.
    prior = (
        np.log1p(-q)
        - np.log(q)
    )

    # Avoid infinite LLRs at p = 0.
    if p == 0.0:

        magnitude = 30.0 + abs(prior)

    else:

        magnitude = (
            np.log1p(-p)
            - np.log(p)
        )

    # LLR convention:
    #
    # Positive -> bit 0
    # Negative -> bit 1

    llrs = (
        (1.0 - 2.0 * y.astype(np.float32))
        * magnitude
        + prior
    )

    return llrs.astype(np.float32)


def make_scrambling(
    batch,
    n,
    info,
    rng,
    enabled,
):
    """Generate random information masks and their codewords."""

    k = len(info)

    if not enabled:

        masks = np.zeros(
            (batch, k),
            dtype=np.uint8,
        )

        return masks, None

    # Random information-bit masks.

    masks = rng.integers(
        0,
        2,
        size=(batch, k),
        dtype=np.uint8,
    )

    # Insert the information masks into zero-frozen
    # polar input vectors.

    u = np.zeros(
        (batch, n),
        dtype=np.uint8,
    )

    u[:, info] = masks

    # Encode the masks into valid polar codewords.

    mask_codeword = polar_transform_bits(u)

    return masks, mask_codeword


# ============================================================
# 6. Main Monte Carlo simulation
# ============================================================

def simulate(
    cfg: SimConfig,
    *,
    batch_limit=1_000_000,
):

    cfg.validate()

    if not isinstance(batch_limit, int) or batch_limit < 1:
        raise ValueError(
            "batch_limit must be a positive integer."
        )

    n = 1 << cfg.n_log2
    k = cfg.key_length
    q = cfg.puf_one_probability

    schemes = get_schemes(cfg)
    decoders = get_decoders(cfg)

    # Python/TensorFlow requires substantially more memory
    # than the native per-thread AFF3CT decoder.
    #
    # Limit the batch size to avoid excessive allocations.

    batch_size = min(
        cfg.batch_size,
        batch_limit,
    )

    # --------------------------------------------------------
    # Separate random-number streams
    # --------------------------------------------------------

    seed = np.random.SeedSequence(cfg.seed)

    device_seed, key_seed, noise_seed, mask_seed = (
        seed.spawn(4)
    )

    noise_seeds = noise_seed.spawn(
        len(cfg.p_values)
    )

    mask_seeds = mask_seed.spawn(
        len(cfg.p_values)
    )

    # --------------------------------------------------------
    # Construct code and enroll device
    # --------------------------------------------------------

    frozen, info = construct_polar_code(
        cfg,
        n,
    )

    checker = build_checker(cfg)

    x, devices = enroll_devices(
        cfg,
        n,
        frozen,
        checker,
        schemes,
        np.random.default_rng(device_seed),
        np.random.default_rng(key_seed),
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    out_dir = Path(cfg.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checker_label = (
        f"sha256_{cfg.hash_bits}"
        if cfg.checker.lower() == "hash"
        else cfg.crc_degree.upper()
    )

    source = (
        "all_zero"
        if cfg.all_zero_device
        else (
            "unbiased"
            if q == 0.5
            else "biased"
        )
    )

    q_label = f"{q:.6g}".replace(".", "p")

    results = []

    # --------------------------------------------------------
    # Print simulation configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("PYTHON/TENSORFLOW POLAR SRAM-PUF SIMULATION")

    print("=" * 70)

    print(f"Block length       : {n}")
    print(f"Secret key length  : {k}")
    print(f"Construction       : {cfg.construction}")
    print(f"Device type        : {source}")
    print(f"P(X=1)             : {q}")
    print(f"Enrollment schemes : {schemes}")
    print(f"Decoders           : {decoders}")
    print(f"Verification       : {checker_label}")
    print(f"BSC design p       : {cfg.design_p}")
    print(f"BSC sweep          : {cfg.p_values}")
    print(f"Seed               : {cfg.seed}")
    print(f"Batch size         : {batch_size}")
    print(f"Scrambling         : {cfg.scramble_codeword}")
    print(f"Maximum blocks     : {cfg.max_blocks}")
    print(f"Minimum errors     : {cfg.min_block_errors}")

    print("=" * 70)

    if q != 0.5:
        print(
            "Note: biased-PUF prior baseline; "
            "secrecy is not established."
        )

    # ========================================================
    # 7. Enrollment and decoder combinations
    # ========================================================

    for scheme, device in devices.items():

        for decoder_name in decoders:

            decoder = build_decoder(
                cfg,
                n,
                frozen,
                checker,
                decoder_name,
            )

            filename = (
                f"PY_BSC_N{n}_K{k}_"
                f"{cfg.construction.upper()}_"
                f"{checker_label}_"
                f"{scheme}_{decoder_name}_"
                f"{source}_q{q_label}_"
                f"scr{int(cfg.scramble_codeword)}.csv"
            )

            output_csv = out_dir / filename

            print("\n" + "=" * 70)

            print(f"Enrollment  : {scheme}")
            print(f"Decoder     : {decoder_name}")

            print(
                f"Helper bits : {device.helper.shape[1]}"
            )

            print(f"Output      : {output_csv}")

            print("=" * 70)

            with output_csv.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(file)

                # Preserve the original Python CSV format.
                writer.writerow(["p", "BLER"])

                # ============================================
                # 8. Sweep BSC crossover probabilities
                # ============================================

                for index, p in enumerate(cfg.p_values):

                    p = float(p)

                    # Each decoder/scheme starts with the
                    # same noise and scrambling RNG streams
                    # for this crossover probability.

                    noise_rng = np.random.default_rng(
                        noise_seeds[index]
                    )

                    mask_rng = np.random.default_rng(
                        mask_seeds[index]
                    )

                    total_blocks = 0
                    total_errors = 0

                    while (
                        total_blocks < cfg.max_blocks
                        and total_errors < cfg.min_block_errors
                    ):

                        batch = min(
                            batch_size,
                            cfg.max_blocks - total_blocks,
                        )

                        # ------------------------------------
                        # Generate noisy reads of the same X.
                        # ------------------------------------

                        llrs = bsc_llrs(
                            x,
                            p,
                            q,
                            batch,
                            noise_rng,
                        )

                        # ------------------------------------
                        # Optional random codeword scrambling
                        # ------------------------------------

                        masks, mask_codeword = (
                            make_scrambling(
                                batch,
                                n,
                                info,
                                mask_rng,
                                cfg.scramble_codeword,
                            )
                        )

                        # ------------------------------------
                        # Apply enrollment offset
                        # ------------------------------------
                        #
                        # Y' = Y XOR Z
                        #
                        # Both schemes now become standard
                        # zero-frozen polar decoding problems.

                        shift = device.offset

                        if mask_codeword is not None:

                            shift = (
                                shift ^ mask_codeword
                            )

                        llrs *= (
                            1.0
                            - 2.0 * shift.astype(np.float32)
                        )

                        # ------------------------------------
                        # Polar decoding
                        # ------------------------------------

                        estimated_key = decode_batch(
                            decoder,
                            decoder_name,
                            llrs,
                            device.tag,
                            info,
                            masks,
                        )

                        # ------------------------------------
                        # Count key reconstruction errors
                        # ------------------------------------

                        block_errors = np.any(
                            estimated_key != device.key,
                            axis=1,
                        )

                        total_errors += int(
                            np.count_nonzero(block_errors)
                        )

                        total_blocks += batch

                        bler = (
                            total_errors / total_blocks
                        )

                        print(
                            f"p: {p:.3f} | "
                            f"BLER: {bler:.6e} | "
                            f"BLE: {total_errors:5d} | "
                            f"BLKs: {total_blocks:12d}",
                            flush=True,
                        )

                    # ----------------------------------------
                    # Final BLER estimate
                    # ----------------------------------------

                    bler = (
                        total_errors / total_blocks
                    )

                    writer.writerow(
                        [
                            f"{p:.6f}",
                            f"{bler:.10e}",
                        ]
                    )

                    file.flush()

                    results.append(
                        {
                            "scheme": scheme,
                            "decoder": decoder_name,
                            "p": p,
                            "BLER": bler,
                            "blocks": total_blocks,
                            "failures": total_errors,
                        }
                    )

            print(f"\nResults saved to: {output_csv}")

    return results


# ============================================================
# 9. Run simulation
# ============================================================

if __name__ == "__main__":

    simulate(SimConfig())
