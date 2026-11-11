
"""Generator-matrix CRC encoder and decoder for TensorFlow/Keras.

CRCEncoder appends polynomial-remainder parity bits to the last axis.
CRCDecoder checks those bits and returns (information_bits, valid_mask).
Inputs must contain binary bits (0 or 1), with shape [..., K] and rank >= 2.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


# ============================================================
# 1. CRC polynomial definitions
# ============================================================

_CRC_SPECS = {
    "CRC24A": [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0],
    "CRC24B": [24, 23, 6, 5, 1, 0],
    "CRC24C": [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0],
    "CRC16": [16, 12, 5, 0],
    "CRC11": [11, 10, 9, 5, 0],
    "CRC6": [6, 5, 0],
    "CRC8": [8, 4, 3, 1, 0],
    "CRC8DVBS2": [8, 7, 6, 4, 2, 0],
}


def _crc_polynomial_bits(
    crc_degree: str,
) -> tuple[np.ndarray, int]:
    """Return polynomial coefficients (MSB first) and CRC length."""

    if crc_degree not in _CRC_SPECS:
        raise ValueError(
            f"Unknown CRC polynomial {crc_degree!r}. "
            f"Choose from {sorted(_CRC_SPECS)}."
        )

    exponents = _CRC_SPECS[crc_degree]
    crc_len = max(exponents)

    polynomial = np.zeros(
        crc_len + 1,
        dtype=np.uint8,
    )

    polynomial[exponents] = 1

    return polynomial[::-1].copy(), crc_len


# ============================================================
# 2. CRC generator matrix
# ============================================================

def _gen_crc_matrix(
    k: int,
    pol_msb: np.ndarray,
) -> np.ndarray:
    """Return the binary parity matrix of shape (K, crc_len)."""

    if k < 1:
        raise ValueError(
            "Information length K must be positive."
        )

    crc_len = len(pol_msb) - 1

    g = np.zeros(
        (k, crc_len),
        dtype=np.uint8,
    )

    # Remainder register, MSB first.
    remainder = np.zeros(
        crc_len,
        dtype=np.uint8,
    )

    remainder[0] = 1

    for i in range(k):

        extended = np.concatenate(
            (remainder, [0])
        ).astype(np.uint8)

        if extended[0]:
            extended ^= pol_msb

        remainder = extended[1:]

        g[k - 1 - i] = remainder

    return g


# ============================================================
# 3. CRC encoder
# ============================================================

class CRCEncoder(Layer):
    """Append CRC parity bits: [..., K] -> [..., K + crc_len]."""

    def __init__(
        self,
        crc_degree: str,
        dtype: tf.DType = tf.float32,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            **kwargs,
        )

        self._crc_degree = crc_degree

        self._crc_pol_msb, self._crc_len = (
            _crc_polynomial_bits(crc_degree)
        )

        self._k = None
        self._n = None
        self._g_mat_crc = None

    @property
    def crc_degree(self) -> str:
        return self._crc_degree

    @property
    def crc_length(self) -> int:
        return self._crc_len

    @property
    def crc_pol(self) -> np.ndarray:
        return self._crc_pol_msb.copy()

    @property
    def k(self) -> int | None:
        return self._k

    @property
    def n(self) -> int | None:
        return self._n

    # --------------------------------------------------------
    # Build the generator matrix once
    # --------------------------------------------------------

    def build(self, input_shape):

        k = input_shape[-1]

        if k is None or int(k) < 1:
            raise ValueError(
                "CRCEncoder needs a fixed, "
                "positive last dimension K."
            )

        self._k = int(k)
        self._n = self._k + self._crc_len

        g = _gen_crc_matrix(
            self._k,
            self._crc_pol_msb,
        )

        self._g_mat_crc = tf.constant(
            g,
            dtype=tf.float32,
        )

        super().build(input_shape)

    # --------------------------------------------------------
    # Append CRC parity bits
    # --------------------------------------------------------

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        tf.debugging.assert_greater_equal(
            tf.rank(inputs),
            2,
        )

        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            self._k,
            message=(
                "CRCEncoder requires the information "
                "length used at build time."
            ),
        )

        # Compute parity = (inputs @ G) mod 2.
        sums = tf.linalg.matmul(
            tf.cast(inputs, tf.float32),
            self._g_mat_crc,
        )

        parity = tf.math.floormod(
            tf.cast(sums, tf.int32),
            2,
        )

        return tf.concat(
            [
                tf.cast(inputs, self.compute_dtype),
                tf.cast(parity, self.compute_dtype),
            ],
            axis=-1,
        )

    def get_config(self):

        config = super().get_config()

        config["crc_degree"] = self._crc_degree

        return config


# ============================================================
# 4. CRC decoder
# ============================================================

class CRCDecoder(Layer):
    """Verify CRC and return (information bits, valid flag)."""

    def __init__(
        self,
        crc_encoder: CRCEncoder,
        dtype: tf.DType = tf.float32,
        **kwargs,
    ):
        if not isinstance(crc_encoder, CRCEncoder):
            raise TypeError(
                "crc_encoder must be a CRCEncoder instance."
            )

        if dtype is None:
            dtype = crc_encoder.dtype

        super().__init__(
            dtype=dtype,
            **kwargs,
        )

        self._encoder = crc_encoder

    @property
    def crc_degree(self) -> str:
        return self._encoder.crc_degree

    @property
    def encoder(self) -> CRCEncoder:
        return self._encoder

    # --------------------------------------------------------
    # Verify and remove CRC parity bits
    # --------------------------------------------------------

    def call(
        self,
        inputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:

        tf.debugging.assert_greater_equal(
            tf.rank(inputs),
            2,
        )

        crc_len = self._encoder.crc_length

        tf.debugging.assert_greater(
            tf.shape(inputs)[-1],
            crc_len,
            message=(
                "CRC word must include information "
                "and parity bits."
            ),
        )

        # Separate information and received parity.
        information = inputs[..., :-crc_len]
        received_parity = inputs[..., -crc_len:]

        # Recalculate expected parity.
        expected_parity = self._encoder(
            information
        )[..., -crc_len:]

        # Valid when all received parity bits match.
        valid = tf.reduce_all(
            tf.equal(
                tf.cast(
                    received_parity,
                    expected_parity.dtype,
                ),
                expected_parity,
            ),
            axis=-1,
            keepdims=True,
        )

        return (
            tf.cast(information, self.compute_dtype),
            valid,
        )

if __name__ == "__main__":
    encoder = CRCEncoder("CRC16")
    decoder = CRCDecoder(encoder)

    # Two information words, K=8.
    bits = tf.constant(
        [
            [1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1],
        ],
        dtype=tf.float32,
    )

    encoded = encoder(bits)

    decoded, valid = decoder(encoded)

    print("Encoded shape:", encoded.shape)
    print("Decoded bits:", decoded.numpy())
    print("CRC valid:", valid.numpy())
