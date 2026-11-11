
"""TensorFlow polar SC and order-one DSCF decoders.

LLR convention (default): positive favors bit 0, negative favors bit 1.
If input_is_logit=True, inputs are assumed to be log(P(1)/P(0))
and are negated to obtain LLRs.

The DSCF decoder returns (full_decoded_u, leaf_llrs), both [B, N].
It uses a supplied hash/CRC checker, or a legacy CRCDecoder, to choose
between SC and single-bit-flip candidates. This is an eager-only reference
implementation; native AFF3CT decoding is used for large simulations.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


# ============================================================
# 1. Successive Cancellation decoder
# ============================================================

class PolarSCDecoder(tf.keras.layers.Layer):
    """SC decoder with optional single-bit flips and frozen-bit values."""

    def __init__(
        self,
        frozen_pos,
        n,
        output_dtype=tf.float32,
        use_fast_sc=False,
        use_flip_sc=False,
        input_is_logit=False,
        tie_breaking="one",
        llr_max=30.0,
        **kwargs,
    ):
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError("output_dtype must be float16, float32 or float64.")

        super().__init__(dtype=output_dtype, **kwargs)

        if not isinstance(n, (int, np.integer)) or n < 2 or n & (n - 1):
            raise ValueError("n must be a power of two >= 2.")

        frozen_pos = np.asarray(frozen_pos)

        if frozen_pos.ndim != 1 or frozen_pos.dtype.kind not in "iu":
            raise TypeError("frozen_pos must be a 1-D integer array.")

        if np.any((frozen_pos < 0) | (frozen_pos >= n)):
            raise ValueError("Frozen positions must be within [0, n).")

        if np.unique(frozen_pos).size != frozen_pos.size:
            raise ValueError("Duplicate frozen positions are not allowed.")

        if frozen_pos.size >= n:
            raise ValueError("At least one information bit is required.")

        if tie_breaking not in ("one", "zero", "random"):
            raise ValueError("tie_breaking must be 'one', 'zero', or 'random'.")

        if not np.isfinite(llr_max) or llr_max <= 0:
            raise ValueError("llr_max must be finite and positive.")

        self._n = int(n)
        self._frozen_pos = np.sort(frozen_pos.astype(np.int32))

        self._info_pos = np.setdiff1d(
            np.arange(n, dtype=np.int32),
            self._frozen_pos,
        )

        self._k = int(self._info_pos.size)

        self._frozen_ind = np.zeros(n, dtype=np.uint8)
        self._frozen_ind[self._frozen_pos] = 1

        self._output_dtype = output_dtype
        self._use_fast_sc = bool(use_fast_sc)
        self._use_flip_sc = bool(use_flip_sc)
        self._input_is_logit = bool(input_is_logit)
        self._tie_breaking = tie_breaking
        self._llr_max = float(llr_max)

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def frozen_pos(self):
        return self._frozen_pos.copy()

    @property
    def info_pos(self):
        return self._info_pos.copy()

    @property
    def llr_max(self):
        return self._llr_max

    @property
    def output_dtype(self):
        return self._output_dtype

    # --------------------------------------------------------
    # Check-node and variable-node updates
    # --------------------------------------------------------

    def _cn_op_tf(self, x, y):
        """Numerically stable exact boxplus check-node update."""

        x = tf.clip_by_value(
            x, -self._llr_max, self._llr_max
        )

        y = tf.clip_by_value(
            y, -self._llr_max, self._llr_max
        )

        return tf.math.softplus(x + y) - tf.reduce_logsumexp(
            tf.stack([x, y], axis=0),
            axis=0,
        )

    @staticmethod
    def _vn_op_tf(x, y, u_hat):
        """Variable-node update."""

        return (1.0 - 2.0 * u_hat) * x + y

    # --------------------------------------------------------
    # Hard decision
    # --------------------------------------------------------

    def _hard_decision(self, llr):
        """Decide 0 for positive LLR, 1 for negative LLR."""

        decided = tf.cast(llr < 0.0, tf.float32)

        if self._tie_breaking == "one":
            return tf.where(llr == 0.0, 1.0, decided)

        if self._tie_breaking == "zero":
            return decided

        random_bits = tf.cast(
            tf.random.uniform(
                tf.shape(llr),
                maxval=2,
                dtype=tf.int32,
            ),
            tf.float32,
        )

        return tf.where(llr == 0.0, random_bits, decided)

    # --------------------------------------------------------
    # Recursive SC decoding
    # --------------------------------------------------------

    def _polar_decode_sc_tf(
        self,
        llr,
        frozen_ind,
        flip_bits=None,
        oracle_dec=False,
        frozen_values=None,
    ):
        """Return (decoded_u, leaf_llrs, upward_partial_sums)."""

        n = len(frozen_ind)

        # Leaf node.
        if n == 1:

            if frozen_ind[0]:
                u = (
                    tf.zeros_like(llr)
                    if frozen_values is None
                    else tf.cast(frozen_values, tf.float32)
                )
                return u, llr, u

            u = self._hard_decision(llr)
            leaf_llr = llr

            if self._use_flip_sc and flip_bits is not None:
                flip = tf.equal(flip_bits, 1.0)
                u = tf.where(flip, 1.0 - u, u)
                leaf_llr = tf.where(flip, -llr, llr)

            return u, leaf_llr, u

        # Optional fast decoding for all-frozen subtrees.
        if (
            self._use_fast_sc
            and frozen_values is None
            and np.all(frozen_ind)
        ):
            zeros = tf.zeros_like(llr)
            return zeros, llr, zeros

        half = n // 2

        a = llr[..., :half]
        b = llr[..., half:]

        frozen_a = frozen_ind[:half]
        frozen_b = frozen_ind[half:]

        flip_a = (
            None if flip_bits is None
            else flip_bits[..., :half]
        )
        flip_b = (
            None if flip_bits is None
            else flip_bits[..., half:]
        )

        values_a = (
            None if frozen_values is None
            else frozen_values[..., :half]
        )
        values_b = (
            None if frozen_values is None
            else frozen_values[..., half:]
        )

        # Upper branch.
        u1, leaf1, partial1 = self._polar_decode_sc_tf(
            self._cn_op_tf(a, b),
            frozen_a,
            flip_a,
            oracle_dec,
            values_a,
        )

        # Lower branch.
        vn_bits = (
            tf.zeros_like(partial1)
            if oracle_dec
            else partial1
        )

        u2, leaf2, partial2 = self._polar_decode_sc_tf(
            self._vn_op_tf(a, b, vn_bits),
            frozen_b,
            flip_b,
            oracle_dec,
            values_b,
        )

        # Re-encode partial sums.
        partial_xor = tf.bitwise.bitwise_xor(
            tf.cast(partial1, tf.int32),
            tf.cast(partial2, tf.int32),
        )

        upward = tf.concat(
            [
                tf.cast(partial_xor, tf.float32),
                partial2,
            ],
            axis=-1,
        )

        return (
            tf.concat([u1, u2], axis=-1),
            tf.concat([leaf1, leaf2], axis=-1),
            upward,
        )

    # --------------------------------------------------------
    # Keras methods
    # --------------------------------------------------------

    def build(self, input_shape):
        if len(input_shape) < 2 or input_shape[-1] != self._n:
            raise ValueError(
                f"Input must have shape [..., {self._n}]."
            )

        super().build(input_shape)

    def call(
        self,
        inputs,
        flip_bits=None,
        oracle_dec=False,
        frozen_values=None,
    ):
        llr = tf.cast(inputs, tf.float32)

        tf.debugging.assert_greater_equal(tf.rank(llr), 2)
        tf.debugging.assert_equal(tf.shape(llr)[-1], self._n)

        llr = tf.reshape(llr, [-1, self._n])

        if self._input_is_logit:
            llr = -llr

        # Optional flip mask.
        flips = None

        if self._use_flip_sc and flip_bits is not None:
            if not (
                isinstance(flip_bits, (list, tuple))
                and len(flip_bits) == 0
            ):
                flips = tf.cast(flip_bits, tf.float32)

                tf.debugging.assert_equal(
                    tf.shape(flips)[-1], self._n
                )

                flips = tf.reshape(
                    flips, [-1, self._n]
                )

                flips = tf.broadcast_to(
                    flips, tf.shape(llr)
                )

                flips = tf.cast(flips > 0.5, tf.float32)

        # Optional frozen-bit values.
        values = None

        if frozen_values is not None:
            values = tf.cast(frozen_values, tf.float32)

            tf.debugging.assert_equal(
                tf.shape(values)[-1], self._n
            )

            values = tf.reshape(
                values, [-1, self._n]
            )

            values = tf.broadcast_to(
                values, tf.shape(llr)
            )

        u, leaf_llrs, _ = self._polar_decode_sc_tf(
            llr,
            self._frozen_ind,
            flips,
            bool(oracle_dec),
            values,
        )

        leaf_llrs = tf.clip_by_value(
            leaf_llrs,
            -self._llr_max,
            self._llr_max,
        )

        return (
            tf.cast(u, self._output_dtype),
            tf.cast(leaf_llrs, self._output_dtype),
        )


# ============================================================
# 2. Dynamic SC-Flip decoder (order one)
# ============================================================

class PolarDSCFlipDecoder(tf.keras.Model):
    """Order-one DSCF with hash/CRC verification.

    'omga' is retained for compatibility with existing callers.
    The initial SC result is retained if all flip attempts fail.
    """

    def __init__(
        self,
        frozen_pos,
        n,
        omga=1,
        crc_decoder=None,
        *,
        checker=None,
        alpha=0.3,
        num_trials=2,
        output_dtype=tf.float32,
        use_fast_sc=False,
        use_flip_sc=True,
        input_is_logit=False,
        tie_breaking="one",
        **kwargs,
    ):
        super().__init__(dtype=output_dtype, **kwargs)

        if int(omga) != 1:
            raise NotImplementedError(
                "Only order-one DSCF (omga=1) is supported."
            )

        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                "alpha must be finite and positive."
            )

        if (
            not isinstance(num_trials, (int, np.integer))
            or num_trials < 0
        ):
            raise ValueError(
                "num_trials must be a nonnegative integer."
            )

        if checker is None and crc_decoder is None:
            raise ValueError(
                "Provide checker or crc_decoder for verification."
            )

        if num_trials > 0 and not use_flip_sc:
            raise ValueError(
                "use_flip_sc must be True when num_trials > 0."
            )

        self.n = int(n)
        self.omga = 1
        self.alpha = float(alpha)
        self.num_trials = int(num_trials)

        self._checker = checker
        self._crc_decoder = crc_decoder
        self._output_dtype = output_dtype

        self.decoder = PolarSCDecoder(
            frozen_pos=frozen_pos,
            n=self.n,
            output_dtype=output_dtype,
            use_fast_sc=use_fast_sc,
            use_flip_sc=use_flip_sc,
            input_is_logit=input_is_logit,
            tie_breaking=tie_breaking,
        )

        self._info_bits = self.decoder.info_pos

    # --------------------------------------------------------
    # Hash/CRC verification
    # --------------------------------------------------------

    def _check(self, candidates, expected_tags):
        """Return one Boolean verification result per candidate."""

        if self._checker is not None:

            if expected_tags is None:
                raise ValueError(
                    "expected_tags are required for PUF verification."
                )

            valid = self._checker.verify(
                np.asarray(candidates, dtype=np.uint8),
                expected_tags,
            )

        else:
            _, valid = self._crc_decoder(
                tf.convert_to_tensor(
                    candidates,
                    dtype=tf.float32,
                )
            )

            valid = valid.numpy()

        valid = np.asarray(
            valid, dtype=bool
        ).reshape(-1)

        if valid.size != len(candidates):
            raise ValueError(
                "Checker must return one Boolean per candidate."
            )

        return valid

    # --------------------------------------------------------
    # DSCF metric
    # --------------------------------------------------------

    def _init_metric(self, leaf_llrs):
        """Log-domain order-one DSCF metric; larger is preferred."""

        magnitudes = tf.gather(
            tf.abs(leaf_llrs),
            self._info_bits,
            axis=1,
        )

        scaled = (
            tf.cast(self.alpha, tf.float32)
            * magnitudes
        )

        return (
            -scaled
            - tf.cumsum(
                tf.math.softplus(-scaled),
                axis=1,
            )
        )

    # --------------------------------------------------------
    # DSCF decoding
    # --------------------------------------------------------

    def call(
        self,
        x,
        frozen_values=None,
        expected_tags=None,
    ):
        if not tf.executing_eagerly():
            raise RuntimeError(
                "DSCF verification requires eager execution."
            )

        llrs = tf.reshape(
            tf.cast(x, tf.float32),
            [-1, self.n],
        )

        values = None

        if frozen_values is not None:

            values = tf.cast(
                frozen_values, tf.float32
            )

            tf.debugging.assert_equal(
                tf.shape(values)[-1],
                self.n,
            )

            values = tf.reshape(
                values, [-1, self.n]
            )

            values = tf.broadcast_to(
                values, tf.shape(llrs)
            )

        tags = (
            expected_tags.numpy()
            if tf.is_tensor(expected_tags)
            else expected_tags
        )

        if tags is not None:

            tags = np.asarray(tags)

            if tags.ndim == 1:
                tags = tags[None, :]

            if tags.shape[0] != int(tf.shape(llrs)[0]):
                raise ValueError(
                    "expected_tags must have one row per word."
                )

        # Initial SC decoding.
        u0, leaf0 = self.decoder(
            llrs,
            frozen_values=values,
        )

        out_u = u0.numpy().copy()
        out_leaf = leaf0.numpy().copy()

        # Check initial candidates.
        passed = self._check(
            out_u[:, self._info_bits],
            tags,
        )

        failed = np.flatnonzero(~passed)

        # ----------------------------------------------------
        # DSCF retry attempts
        # ----------------------------------------------------

        if failed.size and self.num_trials:

            metrics = self._init_metric(leaf0).numpy()

            ranking = np.argsort(
                -metrics,
                axis=1,
                kind="stable",
            )

            flip_positions = self._info_bits[ranking]

            for trial in range(
                min(
                    self.num_trials,
                    self._info_bits.size,
                )
            ):
                if failed.size == 0:
                    break

                positions = flip_positions[
                    failed, trial
                ]

                flip_mask = tf.one_hot(
                    positions,
                    depth=self.n,
                    dtype=tf.float32,
                )

                selected_llrs = tf.gather(
                    llrs,
                    failed,
                    axis=0,
                )

                selected_values = (
                    None
                    if values is None
                    else tf.gather(
                        values, failed, axis=0
                    )
                )

                u_try, leaf_try = self.decoder(
                    selected_llrs,
                    flip_bits=flip_mask,
                    frozen_values=selected_values,
                )

                candidate_u = u_try.numpy()

                candidate_tags = (
                    None
                    if tags is None
                    else tags[failed]
                )

                valid = self._check(
                    candidate_u[:, self._info_bits],
                    candidate_tags,
                )

                if np.any(valid):

                    accepted = failed[valid]

                    out_u[accepted] = candidate_u[valid]

                    out_leaf[accepted] = (
                        leaf_try.numpy()[valid]
                    )

                failed = failed[~valid]

        # Preserve the initial SC result for unsuccessful
        # retries.
        return (
            tf.convert_to_tensor(
                out_u, dtype=self._output_dtype
            ),
            tf.convert_to_tensor(
                out_leaf, dtype=self._output_dtype
            ),
        )
