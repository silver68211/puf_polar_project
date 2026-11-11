
"""
Configuration for polar SRAM-PUF key reconstruction.

Supported enrollment schemes:
    syndrome    - PUF-derived key (Chen et al., 2017)
    code_offset - Independently generated secret key

Supported decoders:
    SC
    DSCF
    SCL

Channel:
    Binary Symmetric Channel (BSC)

Simulation:
    Native C++ / AFF3CT Monte Carlo

Notes:
    - Settings are compatible with dataclasses.replace().
    - A fixed random seed enables reproducible comparisons.
    - Biased PUFs require separate security/leakage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:

    # ==========================================================
    # 1. POLAR CODE PARAMETERS
    # ==========================================================

    # N = 2^n_log2
    # For example:
    # n_log2 = 8  -> N = 256
    # n_log2 = 10 -> N = 1024

    n_log2: int = 10

    # Number of information bits.
    key_length: int = 128

    # Supported:
    # "5G"  -> 5G reliability sequence
    # "BSC" -> Bhattacharyya-based construction
    #
    # Custom "ours" construction can be enabled after
    

    construction: str = "ours"

    # Number of parallel native C++ workers.
    # For HPC runs, Slurm can override this value.

    num_threads: int = 32

    # ==========================================================
    # 2. SRAM-PUF PARAMETERS
    # ==========================================================

    # Probability that a physical SRAM cell contains 1.
    #
    # 0.5 -> Unbiased PUF
    # 0.4 -> Biased PUF

    puf_one_probability: float = 0.5

    # Use a fixed all-zero physical PUF response.
    # Only valid for the unbiased baseline (q=0.5).

    all_zero_device: bool = False

    # Apply a random codeword mask during simulation.
    # The mask is removed before key verification.

    scramble_codeword: bool = True

    # ==========================================================
    # 3. ENROLLMENT SCHEMES
    # ==========================================================

    # "syndrome":
    #
    #   U = X G_N
    #   S = U_A
    #   W = U_F
    #
    #   Key is extracted from the PUF.
    #   Public helper contains N-K bits.
    #
    # "code_offset":
    #
    #   S is generated independently.
    #   C = encode(S)
    #   W = X XOR C
    #
    #   Public helper contains N bits.
    #
    # "transform" is also accepted as an alias
    # for "syndrome".
    #
    # Select either or both schemes:
    #
    # ("syndrome",)
    # ("code_offset",)
    # ("syndrome", "code_offset")

    enrollment_schemes: tuple[str, ...] = (
        "syndrome",
        "code_offset",
    )

    # ==========================================================
    # 4. BSC CHANNEL PARAMETERS
    # ==========================================================

    # BSC crossover probability used for polar construction.
    # Relevant when construction="BSC".

    design_p: float = 0.15

    # Crossover probabilities used in Monte Carlo.
    # The polar construction remains fixed across all p.

    p_values: tuple[float, ...] = ( 
        0.20, 
        0.22, 
        0.24, 
        0.26, 
        0.28, 
        0.30, 
        0.32,
    )

    # ==========================================================
    # 5. DECODER PARAMETERS
    # ==========================================================

    # Choose any combination:
    #
    # ("SC",)
    # ("DSCF",)
    # ("SCL",)
    # ("SC", "DSCF")
    # ("DSCF", "SCL")
    # ("SC", "DSCF", "SCL")
    #
    # Decoder order determines execution order.

    decoder: tuple[str, ...] = (
        "SC", "DSCF"
    )

    # DSCF: Maximum number of additional flip attempts.
    # Total decoding passes are at most 1 + num_trials.

    num_trials: int = 10

    # DSCF flip-metric parameter.

    alpha: float = 0.3367

    # SCL list sizes.
    # One independent simulation is performed per L.

    list_sizes: tuple[int, ...] = (
        2,
        4,
        8,
        16,
    )

    # ==========================================================
    # 6. RANDOM NUMBER GENERATION
    # ==========================================================

    # Fixed seed for reproducibility.
    #
    # This is especially useful when comparing:
    #   - syndrome vs. code_offset
    #   - SC vs. DSCF vs. SCL
    #   - Different HPC nodes
    #
    # This is a simulation seed, not a cryptographic
    # random-number generator.

    seed: int = 2026

    # ==========================================================
    # 7. KEY VERIFICATION
    # ==========================================================

    # Native AFF3CT engine supports:
    #
    # "hash" -> SHA-256 verification
    # "crc"  -> CRC16 verification

    checker: str = "hash"

    hash_name: str = "sha256"

    # SHA-256 tag length in bits.
    # Must be a multiple of 8, between 8 and 256.

    hash_bits: int = 128

    crc_degree: str = "CRC16"

    # ==========================================================
    # 8. MONTE CARLO PARAMETERS
    # ==========================================================

    # Legacy Python batch-size parameter.
    #
    # The native C++ Monte Carlo engine currently manages
    # frames internally rather than using this batch size.

    batch_size: int = 10_000

    # Stop each BSC probability after this many
    # incorrectly reconstructed keys, unless max_blocks
    # is reached first.

    min_block_errors: int = 100

    # Maximum number of frames for each p.

    max_blocks: int = 10_000_000_000

    # ==========================================================
    # 9. OUTPUT
    # ==========================================================

    # Base directory for CSV results.
    #
    # The simulation driver will create separate files
    # for each enrollment scheme and decoder combination.

    out_dir: str = "results"

    # ==========================================================
    # 10. AUTOMATIC VALIDATION
    # ==========================================================

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:

        # ------------------------------------------------------
        # Polar-code parameters
        # ------------------------------------------------------

        if self.n_log2 < 1:
            raise ValueError(
                "n_log2 must be positive."
            )

        n = 1 << self.n_log2

        if not (0 < self.key_length < n):
            raise ValueError(
                "Require 0 < key_length < N."
            )

        construction = self.construction.strip().upper()

        if construction not in (
            "5G",
            "BSC",
            "BHATTACHARYYA",
            'OURS'
        ):
            raise ValueError(
                "construction must be '5G' or 'BSC'. "
                "Custom constructions require additional "
                "implementation in sim_aff3ct.py."
            )

        if self.num_threads < 1:
            raise ValueError(
                "num_threads must be >= 1."
            )

        # ------------------------------------------------------
        # SRAM-PUF parameters
        # ------------------------------------------------------

        if not (
            0 < self.puf_one_probability < 1
        ):
            raise ValueError(
                "puf_one_probability must be in (0, 1)."
            )

        if (
            self.all_zero_device
            and self.puf_one_probability != 0.5
        ):
            raise ValueError(
                "All-zero baseline requires "
                "puf_one_probability = 0.5."
            )

        # ------------------------------------------------------
        # Enrollment scheme selection
        # ------------------------------------------------------

        schemes = self.enrollment_schemes

        if isinstance(schemes, str):
            schemes = (schemes,)

        schemes = tuple(
            str(s).strip().lower()
            for s in schemes
        )

        if not schemes:
            raise ValueError(
                "At least one enrollment scheme "
                "must be selected."
            )

        normalized_schemes = tuple(
            "syndrome" if s == "transform" else s
            for s in schemes
        )

        allowed_schemes = {
            "syndrome",
            "code_offset",
        }

        if any(
            scheme not in allowed_schemes
            for scheme in normalized_schemes
        ):
            raise ValueError(
                "Enrollment schemes must be "
                "'syndrome' or 'code_offset'."
            )

        if len(set(normalized_schemes)) != len(
            normalized_schemes
        ):
            raise ValueError(
                "Duplicate enrollment schemes "
                "are not allowed."
            )

        # Code-offset enrollment is also allowed for
        # biased PUFs. Security and leakage under bias
        # must be evaluated separately.

        # ------------------------------------------------------
        # BSC parameters
        # ------------------------------------------------------

        if not (
            0 < self.design_p < 0.5
        ):
            raise ValueError(
                "design_p must be in (0, 0.5)."
            )

        if not self.p_values:
            raise ValueError(
                "p_values cannot be empty."
            )

        for p in self.p_values:

            if not (0 <= p < 0.5):
                raise ValueError(
                    "Every BSC crossover probability "
                    "must be in [0, 0.5)."
                )

        if len(set(self.p_values)) != len(
            self.p_values
        ):
            raise ValueError(
                "Duplicate p_values are not allowed."
            )

        # ------------------------------------------------------
        # Decoder selection
        # ------------------------------------------------------

        decoders = self.decoder

        if isinstance(decoders, str):
            decoders = (decoders,)

        decoders = tuple(
            str(name).strip().upper()
            for name in decoders
        )

        if not decoders:
            raise ValueError(
                "At least one decoder must be selected."
            )

        allowed_decoders = {
            "SC",
            "DSCF",
            "SCL",
        }

        if any(
            name not in allowed_decoders
            for name in decoders
        ):
            raise ValueError(
                "Decoder must be SC, DSCF, or SCL."
            )

        if len(set(decoders)) != len(decoders):
            raise ValueError(
                "Duplicate decoder selections "
                "are not allowed."
            )

        # ------------------------------------------------------
        # DSCF parameters
        # ------------------------------------------------------

        if self.num_trials < 0:
            raise ValueError(
                "num_trials must be nonnegative."
            )

        if self.alpha <= 0:
            raise ValueError(
                "alpha must be positive."
            )

        # ------------------------------------------------------
        # SCL list sizes
        # ------------------------------------------------------

        if "SCL" in decoders:

            if not self.list_sizes:
                raise ValueError(
                    "SCL list_sizes cannot be empty."
                )

            if (
                len(set(self.list_sizes))
                != len(self.list_sizes)
            ):
                raise ValueError(
                    "Duplicate SCL list sizes "
                    "are not allowed."
                )

            for L in self.list_sizes:

                if (
                    not isinstance(L, int)
                    or L < 1
                    or (L & (L - 1)) != 0
                ):
                    raise ValueError(
                        "SCL list sizes must be "
                        "positive powers of two."
                    )

        # ------------------------------------------------------
        # Key verification
        # ------------------------------------------------------

        checker = self.checker.strip().lower()

        if checker not in (
            "hash",
            "crc",
        ):
            raise ValueError(
                "Native Monte Carlo supports "
                "'hash' or 'crc' verification."
            )

        if checker == "hash":

            if self.hash_name.strip().lower() != "sha256":
                raise ValueError(
                    "Native checker supports SHA-256 only."
                )

            if not (
                8 <= self.hash_bits <= 256
                and self.hash_bits % 8 == 0
            ):
                raise ValueError(
                    "hash_bits must be byte-aligned "
                    "and between 8 and 256."
                )

        elif checker == "crc":

            if self.crc_degree.strip().upper() != "CRC16":
                raise ValueError(
                    "Native checker supports CRC16 only."
                )

        # ------------------------------------------------------
        # Random seed
        # ------------------------------------------------------

        if not (
            0 <= self.seed < (1 << 64)
        ):
            raise ValueError(
                "seed must be an unsigned 64-bit integer."
            )

        # ------------------------------------------------------
        # Monte Carlo stopping conditions
        # ------------------------------------------------------

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.min_block_errors < 1:
            raise ValueError(
                "min_block_errors must be positive."
            )

        if self.max_blocks < 1:
            raise ValueError(
                "max_blocks must be positive."
            )

        # ------------------------------------------------------
        # Output directory
        # ------------------------------------------------------

        if not (
            isinstance(self.out_dir, str)
            and self.out_dir.strip()
        ):
            raise ValueError(
                "out_dir must be a nonempty string."
            )
