from dataclasses import dataclass
import numpy as np
import os


@dataclass(frozen=True)
class SimConfig:
    # ==========================================================
    # 1. POLAR CODE PARAMETERS
    # ==========================================================

    n_log2: int = 10  # N = 2^8 = 256
    key_length: int = 128  # Secret key length K
    construction: str = "5G"  # "5G" or "BSC"

    # Number of parallel C++ simulation workers.
    # Each worker has its own AFF3CT SC decoder.
    num_threads: int = 3

    # ==========================================================
    # 2. SRAM-PUF PARAMETERS
    # ==========================================================

    # Probability that an SRAM cell powers up as 1.
    #
    # 0.5 -> Unbiased SRAM-PUF
    # 0.4 -> Biased SRAM-PUF
    #

    puf_one_probability: float = 0.5

    # If True, use X = [0, 0, ..., 0].
    # This is useful for conventional all-zero polar-code
    # simulations over the BSC.
    #
    # Only use with puf_one_probability = 0.5.

    all_zero_device: bool = False
    scramble_codeword: bool = True

    # ==========================================================
    # 3. BSC CHANNEL PARAMETERS
    # ==========================================================

    # Crossover probability used to design the polar code.
    # The frozen and information positions are calculated
    # once and remain unchanged during the entire simulation.

    design_p: float = 0.15

    # Crossover probabilities used to measure BLER.

    p_values: tuple = (
        
        0.15,
        0.18,
        0.20,
        0.22,
        0.24,
        0.26,
        0.28,
        0.30,
    )

    # ==========================================================
    # 4. DECODER PARAMETERS
    # ==========================================================

    # Available decoders:
    # "SC"   -> Successive Cancellation
    # "DSCF" -> Dynamic Successive Cancellation Flip

    decoder: tuple[str, ...] = ("SC", "DSCF", "SCL")  # "SC", "DSCF", "SCL",  "all"
    
    seed: int = int.from_bytes(
        os.urandom(8), "little"
    )   # int.from_bytes(os.urandom(8), "little")

    # Maximum number of DSCF flip attempts.
    num_trials: int = 10

    # DSCF metric parameter.
    alpha: float = 0.3367

    # SCL parameters
    list_sizes: tuple[int, ...] = (2, 4, 8, 16)

    # ==========================================================
    # 5. KEY VERIFICATION
    # ==========================================================

    # "hash" -> Hash-based verification
    # "crc"  -> CRC verification
    # "none" -> No verification (SC only)

    checker: str = "hash"

    hash_name: str = "sha256"
    hash_bits: int = 128

    crc_degree: str = "CRC16"

    # ==========================================================
    # 6. MONTE CARLO SIMULATION PARAMETERS
    # ==========================================================

    # Number of independent noisy SRAM readings per batch.
    batch_size: int = 1000000

    # Stop simulating a crossover probability after this
    # many incorrectly reconstructed keys.
    min_block_errors: int = 200

    # Maximum number of noisy readings for each p value.
    max_blocks: int = 1000_000_000

    # ==========================================================
    # 7. OPTIONAL COMPARISON
    # ==========================================================

    # Compare syndrome construction with code-offset.
    # This option is restricted to the unbiased case here.

    compare_code_offset: bool = False

    # ==========================================================
    # 8. OUTPUT
    # ==========================================================

    # Directory for CSV results.
    out_dir: str = "results"

    # ==========================================================
    # 9. CONFIGURATION VALIDATION
    # ==========================================================

    def validate(self):

        if self.n_log2 < 1:
            raise ValueError("n_log2 must be positive.")

        N = 2**self.n_log2

        if not 0 < self.key_length < N:
            raise ValueError("Require 0 < key_length < N.")

        if not 0 < self.puf_one_probability < 1:
            raise ValueError("puf_one_probability must be in (0, 1).")

        if self.all_zero_device and self.puf_one_probability != 0.5:
            raise ValueError("All-zero mode requires puf_one_probability = 0.5.")

        if not 0 < self.design_p < 0.5:
            raise ValueError("design_p must be in (0, 0.5).")

        if not self.p_values:
            raise ValueError("p_values cannot be empty.")

        for p in self.p_values:
            if not 0 <= p < 0.5:
                raise ValueError("Every BSC crossover probability must be in [0, 0.5).")

        if self.decoder.upper() not in ("SC", "DSCF"):
            raise ValueError("decoder must be SC or DSCF.")

        if self.checker.lower() not in ("hash", "crc", "none"):
            raise ValueError("Invalid checker.")

        if self.decoder.upper() == "DSCF" and self.checker.lower() == "none":
            raise ValueError("DSCF requires hash or CRC verification.")

        if self.decoder.upper() == "DSCF" and self.num_trials < 0:
            raise ValueError("num_trials cannot be negative.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.min_block_errors <= 0:
            raise ValueError("min_block_errors must be positive.")

        if self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive.")

        if self.compare_code_offset and self.puf_one_probability != 0.5:
            raise ValueError(
                "Code-offset comparison is enabled only for unbiased PUFs."
            )

        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1.")
