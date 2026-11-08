"""
SRAM-PUF KEY GENERATION USING POLAR CODES
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import SimConfig
from DSCFlip import PolarSCDecoder, PolarDSCFlipDecoder
from Enrolment import SRAMPUFEnrollment
from constructions import bhattacharyya_design
from keycheck import KeyChecker
from utils import sample_channel, generate_5g_ranking


# ==============================================================
# 1. POLAR CODE CONSTRUCTION
# ==============================================================


def construct_polar_code(cfg, N):

    # Construct the polar code for the design BSC.
    #
    # frozen: indices containing public helper data
    # info:   indices containing secret-key bits
    #

    # frozen, info, _ = bhattacharyya_design(
    #     N,
    #     cfg.key_length,
    #     channel="BSC",
    #     design_p=cfg.design_p,
    # )

    frozen, info = generate_5g_ranking(cfg.key_length, N)

    return frozen, info


# ==============================================================
# 2. KEY VERIFICATION
# ==============================================================


def build_checker(cfg):

    # Hash and CRC tags are stored externally.
    # They do not increase the polar-code information length.

    return KeyChecker(
        method=cfg.checker.lower(),
        hash_name=cfg.hash_name,
        hash_bits=cfg.hash_bits,
        crc_degree=cfg.crc_degree,
    )


# ==============================================================
# 3. ENROLL ONE SRAM-PUF DEVICE
# ==============================================================


def enroll_device(cfg, N, frozen, checker, rng):

    q = cfg.puf_one_probability

    enrollment = SRAMPUFEnrollment(
        n=N,
        frozen_pos=frozen,
        biased=(q != 0.5),
        prob_one=q,
        checker=checker,
    )

    # Generate exactly ONE SRAM device.
    #
    # The original SRAM values X never change afterward.

    if cfg.all_zero_device:
        X = np.zeros((1, N), dtype=np.uint8)

        device = enrollment.enroll(
            x=X,
            rng=rng,
        )

    else:
        device = enrollment.enroll(
            batch_size=1,
            rng=rng,
        )

    return enrollment, device


# ==============================================================
# 4. BUILD THE POLAR DECODER
# ==============================================================


def build_decoder(cfg, N, frozen, checker):

    if cfg.decoder.upper() == "SC":
        return PolarSCDecoder(
            frozen_pos=frozen,
            n=N,
        )

    if cfg.decoder.upper() == "DSCF":
        return PolarDSCFlipDecoder(
            frozen_pos=frozen,
            n=N,
            omga=1,
            checker=checker,
            alpha=cfg.alpha,
            num_trials=cfg.num_trials,
            use_flip_sc=True,
        )

    raise ValueError("Unsupported decoder.")


# ==============================================================
# 5. DECODING ONE BATCH
# ==============================================================


def decode_batch(
    decoder,
    decoder_name,
    llr,
    frozen_values,
    expected_tag,
    info,
):

    batch_size = llr.shape[0]

    # Each reading belongs to the same device.
    # Therefore the frozen values are identical in every row.

    frozen_batch = np.broadcast_to(
        frozen_values,
        (batch_size, frozen_values.shape[1]),
    )

    frozen_batch = tf.convert_to_tensor(
        frozen_batch,
        dtype=tf.float32,
    )

    llr_tensor = tf.convert_to_tensor(
        llr,
        dtype=tf.float32,
    )

    # SC decoding

    if decoder_name == "SC":
        decoded_full, _ = decoder(
            llr_tensor,
            frozen_values=frozen_batch,
        )

    # DSCF decoding with hash/CRC verification

    else:
        tags = np.broadcast_to(
            expected_tag,
            (batch_size, expected_tag.shape[1]),
        )

        decoded_full, _ = decoder.call(
            llr_tensor,
            frozen_values=frozen_batch,
            expected_tags=tags,
        )

    # Extract the K estimated secret-key bits.

    decoded_full = decoded_full.numpy()

    estimated_key = decoded_full[:, info].astype(np.uint8)

    return estimated_key


# ==============================================================
# 6. MAIN MONTE CARLO SIMULATION
# ==============================================================


def simulate(cfg: SimConfig):

    cfg.validate()

    N = 2**cfg.n_log2
    K = cfg.key_length
    q = cfg.puf_one_probability

    decoder_name = cfg.decoder.upper()

    # No fixed seed.
    #
    # NumPy creates a fresh generator initialized from
    # operating-system entropy.
    #
    # Different executions therefore generate different
    # device samples and independent noise realizations.

    rng = np.random.default_rng()

    # ----------------------------------------------------------
    # STEP 1: Construct polar code
    # ----------------------------------------------------------

    frozen, info = construct_polar_code(cfg, N)

    # ----------------------------------------------------------
    # STEP 2: Create hash/CRC checker
    # ----------------------------------------------------------

    checker = build_checker(cfg)

    # ----------------------------------------------------------
    # STEP 3: Enroll ONE device
    # ----------------------------------------------------------

    enrollment, device = enroll_device(
        cfg,
        N,
        frozen,
        checker,
        rng,
    )

    # Store the enrolled device information.
    #
    # These arrays are never modified during the simulation.

    X = device.x
    S = device.key
    W = device.frozen_full
    tag = device.tag

    # ----------------------------------------------------------
    # STEP 4: Build decoder
    # ----------------------------------------------------------

    decoder = build_decoder(
        cfg,
        N,
        frozen,
        checker,
    )

    # ----------------------------------------------------------
    # STEP 5: Optional code-offset construction
    # ----------------------------------------------------------

    offset_sign = None

    if cfg.compare_code_offset:
        _, W_offset = enrollment.code_offset(device)

        offset_sign = 1.0 - 2.0 * W_offset.astype(np.float32)

    # ----------------------------------------------------------
    # STEP 6: Prepare output CSV
    # ----------------------------------------------------------

    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.all_zero_device:
        source = "all_zero"
    elif q == 0.5:
        source = "unbiased"
    else:
        source = "biased"

    q_string = f"{q:.3f}".replace(".", "p")

    filename = (
        f"BLER_BSC_N{N}_K{K}_"
        f"{decoder_name}_{source}_"
        f"q{q_string}_{cfg.checker.lower()}.csv"
    )

    output_csv = output_dir / filename

    # ----------------------------------------------------------
    # STEP 7: Print simulation settings
    # ----------------------------------------------------------

    print("\n" + "=" * 70)

    print("POLAR SRAM-PUF SIMULATION")

    print("=" * 70)
    print(f"Block length       : {N}")
    print(f"Secret key length  : {K}")
    print(f"Helper data length : {N - K}")
    print(f"Device type        : {source}")
    print(f"P(X=1)             : {q}")
    print(f"Channel            : BSC")
    print(f"Design p           : {cfg.design_p}")
    print(f"Decoder            : {decoder_name}")
    print(f"Verification       : {cfg.checker}")
    print(f"Batch size         : {cfg.batch_size}")
    print(f"Maximum blocks     : {cfg.max_blocks}")
    print(f"Minimum errors     : {cfg.min_block_errors}")
    print("=" * 70)

    if q != 0.5:
        print("Note: biased PUF baseline; secrecy is not guaranteed.")

    # ----------------------------------------------------------
    # STEP 8: Sweep the BSC crossover probabilities
    # ----------------------------------------------------------

    results = []

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        # CSV contains exactly TWO columns.
        writer.writerow(["p", "BLER"])

        for p in cfg.p_values:
            print("=" * 70)
            total_blocks = 0
            total_errors = 0
            offset_errors = 0

            # --------------------------------------------------
            # Repeated noisy reads of the SAME SRAM device
            # --------------------------------------------------

            while total_blocks < cfg.max_blocks and total_errors < cfg.min_block_errors:
                batch = min(
                    cfg.batch_size,
                    cfg.max_blocks - total_blocks,
                )

                # Repeat the same SRAM pattern X.
                #
                # This does NOT generate new SRAM devices.
                #
                # Shape: [batch, N]

                X_batch = np.broadcast_to(
                    X,
                    (batch, N),
                )

                # ------------------------------------------------
                # BSC CHANNEL
                # ------------------------------------------------
                #
                # Each cell is flipped independently with
                # probability p.
                #
                # Y = X XOR E
                #
                # For unbiased PUFs:
                #
                # LLR = (1-2Y) * log((1-p)/p)
                #
                # For biased PUFs, sample_channel also adds
                # the corresponding source-prior LLR.

                _, llr = sample_channel(
                    X_batch,
                    channel="BSC",
                    p=float(p),
                    prob_one=q,
                    rng=rng,
                )

                # ------------------------------------------------
                # POLAR DECODING
                # ------------------------------------------------

                S_hat = decode_batch(
                    decoder,
                    decoder_name,
                    llr,
                    W,
                    tag,
                    info,
                )

                # ------------------------------------------------
                # COUNT KEY RECONSTRUCTION ERRORS
                # ------------------------------------------------
                #
                # A block fails if at least ONE key bit is wrong.

                block_errors = np.any(
                    S_hat != S,
                    axis=1,
                )

                total_errors += int(np.count_nonzero(block_errors))

                # ------------------------------------------------
                # OPTIONAL CODE-OFFSET COMPARISON
                # ------------------------------------------------

                if cfg.compare_code_offset:
                    llr_offset = (llr * offset_sign).astype(np.float32)

                    zero_frozen = np.zeros_like(W)

                    S_offset = decode_batch(
                        decoder,
                        decoder_name,
                        llr_offset,
                        zero_frozen,
                        tag,
                        info,
                    )

                    offset_errors += int(
                        np.count_nonzero(np.any(S_offset != S, axis=1))
                    )

                total_blocks += batch
                BLER = total_errors / total_blocks
                print(
                    f"p: {p:1.3f} |"
                    f"BLER: {BLER:1.6e} |"
                    f"BLE: {total_errors:4d} |"
                    f"BLKs: {total_blocks:12d}"
                )

            # --------------------------------------------------
            # BLER ESTIMATION
            # --------------------------------------------------

            BLER = total_errors / total_blocks

            results.append((float(p), BLER))

            # Save [p, BLER] only.

            writer.writerow(
                [
                    f"{p:.6f}",
                    f"{BLER:.10e}",
                ]
            )

            file.flush()

            if cfg.compare_code_offset:
                print(f"  Code-offset BLER: {offset_errors / total_blocks:.6e}")

    print("=" * 70)
    print(f"\nResults saved to: {output_csv}")

    return results


# ==============================================================
# 7. RUN SIMULATION
# ==============================================================

if __name__ == "__main__":
    # All user-editable parameters are in config.py.

    config = SimConfig()

    simulate(config)
