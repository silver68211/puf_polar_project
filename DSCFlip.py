import numpy as np
import numbers
import tensorflow as tf 






class PolarSCDecoder(tf.keras.layers.Layer):
    """
    Successive Cancellation (SC) decoder for polar codes.

    Inputs
    ------
    inputs : tf.Tensor, shape [..., n]
        Channel LLRs or logits (you negate them below to convert logits->LLR).
    flip_bits : Optional[tf.Tensor], shape [..., n] or [batch, n]
        Binary mask indicating which bit decisions to flip (SC-Flip support).
        If provided, is only applied at leaf decisions.
    oracle_dec : bool
        If True, forces lower-branch VN update to use zeros (oracle-like behavior).

    Returns
    -------
    u_hat_n : tf.Tensor, shape [batch, n]
        Estimated u (including frozen positions).
    llr_out : tf.Tensor, shape [batch, n]
        Leaf LLRs after traversal (clipped).
    """

    def __init__(
        self,
        frozen_pos,
        n,
        output_dtype=tf.float32,
        use_fast_sc=False,
        use_flip_sc=False,
        llr_max=80.0,
        **kwargs,
    ):
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError("output_dtype must be one of {tf.float16, tf.float32, tf.float64}.")
        if output_dtype is not tf.float32:
            tf.print("Note: decoder uses tf.float32 for internal calculations.")

        super().__init__(dtype=output_dtype, **kwargs)
        self._output_dtype = output_dtype

        # ---- validate n
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        if np.log2(n) != int(np.log2(n)):
            raise ValueError("n must be a power of 2.")

        # ---- validate frozen_pos
        frozen_pos = np.asarray(frozen_pos)
        if frozen_pos.dtype.kind not in ("i", "u"):
            raise TypeError("frozen_pos must contain integers.")
        if frozen_pos.ndim != 1:
            raise ValueError("frozen_pos must be a 1-D array of positions.")
        if len(frozen_pos) > n:
            raise ValueError("len(frozen_pos) cannot exceed n.")
        if np.any(frozen_pos < 0) or np.any(frozen_pos >= n):
            raise ValueError("frozen_pos entries must be in [0, n-1].")

        frozen_pos = np.unique(frozen_pos)  # ensure no duplicates

        # ---- store code params
        self._n = n
        self._frozen_pos = frozen_pos
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        if self._k != len(self._info_pos):
            raise RuntimeError("Internal error: invalid info_pos generated.")

        self._llr_max = float(llr_max)

        # frozen indicator vector: 1 for frozen positions, 0 otherwise
        self._frozen_ind = np.zeros(self._n, dtype=np.int8)
        self._frozen_ind[self._frozen_pos] = 1

        self._use_fast_sc = bool(use_fast_sc)
        self._use_flip_sc = bool(use_flip_sc)

    # --------------------
    # Public properties
    # --------------------
    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def frozen_pos(self):
        return self._frozen_pos

    @property
    def info_pos(self):
        return self._info_pos

    @property
    def llr_max(self):
        return self._llr_max

    @property
    def output_dtype(self):
        return self._output_dtype

    # --------------------
    # Internal ops
    # --------------------
    def _clip_llr(self, x):
        return tf.clip_by_value(x, -self._llr_max, self._llr_max)

    def _cn_op_tf(self, x, y):
        """Check-node update (boxplus) for LLRs."""
        x_in = self._clip_llr(x)
        y_in = self._clip_llr(y)

        # stable form:
        # log(1+exp(x+y)) - log(exp(x)+exp(y))
        llr_out = tf.math.softplus(x_in + y_in)
        llr_out -= tf.reduce_logsumexp(tf.stack([x_in, y_in], axis=0), axis=0)
        return llr_out

    def _vn_op_tf(self, x, y, u_hat):
        """Variable-node update for LLRs."""
        return tf.multiply((1.0 - 2.0 * u_hat), x) + y

    def _hard_decision(self, llr):
        """Hard decision: 0 if llr>=0 else 1, with tie-breaking to 1 (matches your code)."""
        u = 0.5 * (1.0 - tf.sign(llr))
        u = tf.where(tf.equal(u, 0.5), tf.ones_like(u), u)
        return u

    def _polar_decode_sc_tf(self, llr_ch, frozen_ind, flip_bits=None, oracle_dec=False, frozen_values=None):
        """
        Recursive SC decoding.
        frozen_ind: numpy array of {0,1} with length n (python-side).
        flip_bits: tf.Tensor with shape [batch, n] or None.
        """
        n = int(len(frozen_ind))

        # Fast SC prune: all frozen
        if n > 1 and frozen_values is None and self._use_fast_sc and int(np.sum(frozen_ind)) == n:
            u_hat = tf.zeros_like(llr_ch)
            return u_hat, llr_ch, u_hat

        if n > 1:
            half = n // 2

            llr1 = llr_ch[..., :half]
            llr2 = llr_ch[..., half:]
            frozen1 = frozen_ind[:half]
            frozen2 = frozen_ind[half:]

            flip1 = None
            flip2 = None
            if self._use_flip_sc and flip_bits is not None:
                flip1 = flip_bits[..., :half]
                flip2 = flip_bits[..., half:]

            fv1 = None if frozen_values is None else frozen_values[..., :half]
            fv2 = None if frozen_values is None else frozen_values[..., half:]

            # upper
            llr_up_in = self._cn_op_tf(llr1, llr2)
            u1, llr_up_out, u1_up = self._polar_decode_sc_tf(llr_up_in, frozen1, flip1, oracle_dec, fv1)

            # lower (VN uses either oracle zeros or previous decisions)
            u_for_vn = tf.zeros_like(u1_up) if oracle_dec else u1_up
            llr_lo_in = self._vn_op_tf(llr1, llr2, u_for_vn)
            u2, llr_lo_out, u2_up = self._polar_decode_sc_tf(llr_lo_in, frozen2, flip2, oracle_dec, fv2)

            # combine decisions
            u_hat = tf.concat([u1, u2], axis=-1)
            llr_out = tf.concat([llr_up_out, llr_lo_out], axis=-1)

            # re-encode partial sums upward:
            # u1_up <- u1_up xor u2_up
            u1_up_i = tf.cast(u1_up, tf.int8)
            u2_up_i = tf.cast(u2_up, tf.int8)
            u1_up_xor = tf.bitwise.bitwise_xor(u1_up_i, u2_up_i)
            u1_up = tf.cast(u1_up_xor, tf.float32)

            u_up = tf.concat([u1_up, u2_up], axis=-1)
            return u_hat, llr_out, u_up

        # ---- leaf
        # frozen_ind is length-1 numpy array; check its only value
        is_frozen = int(frozen_ind[0]) == 1
        llr_leaf = llr_ch  # shape [batch, 1]

        if is_frozen:
            u = (tf.zeros_like(llr_leaf) if frozen_values is None
                 else tf.cast(frozen_values, tf.float32))
            return u, llr_leaf, u

        # info bit
        u = self._hard_decision(llr_leaf)

        if self._use_flip_sc and flip_bits is not None:
            # flip_bits should be shape [batch, 1] here
            flip_bits = tf.cast(flip_bits, u.dtype)
            u = tf.where(tf.equal(flip_bits, 1.0), 1.0 - u, u)
            llr_leaf = tf.where(tf.equal(flip_bits, 1.0), -llr_leaf, llr_leaf)

        return u, llr_leaf, u

    # --------------------
    # Keras methods
    # --------------------
    def build(self, input_shape):
        if input_shape[-1] != self._n:
            raise ValueError(f"Invalid input shape: last dim must be n={self._n}.")
        if len(input_shape) < 2:
            raise ValueError("Inputs must have at least 2 dimensions.")
        super().build(input_shape)

    def call(self, inputs, flip_bits=None, oracle_dec=False, frozen_values=None):
        tf.debugging.assert_type(inputs, self.dtype, "Invalid input dtype.")

        x = tf.cast(inputs, tf.float32)
        tf.debugging.assert_equal(tf.shape(x)[-1], self._n, "Last input dimension must be length n.")
        tf.debugging.assert_greater(tf.rank(x), 1)

        # Flatten batch dims to [batch, n]
        x = tf.reshape(x, [-1, self._n])

        # Convert logits -> LLR if your upstream provides logits
        llr_ch =  x

        fb = None
        if self._use_flip_sc and flip_bits is not None:
            fb = tf.cast(flip_bits, tf.float32)
            fb = tf.reshape(fb, [-1, self._n])
            tf.debugging.assert_equal(tf.shape(fb)[-1], self._n, "flip_bits last dim must be n.")
            # ensure binary-ish
            fb = tf.where(fb > 0.5, tf.ones_like(fb), tf.zeros_like(fb))

        fv = None
        if frozen_values is not None:
            fv = tf.cast(frozen_values, tf.float32)
            fv = tf.reshape(fv, [-1, self._n])
            fv = tf.broadcast_to(fv, tf.shape(llr_ch))
        u_hat_n, llr_out, _ = self._polar_decode_sc_tf(llr_ch, self._frozen_ind, fb, bool(oracle_dec), fv)
        llr_out = self._clip_llr(llr_out)

        # return in internal flattened shape like your original (batch, n)
        return tf.cast(u_hat_n, self._output_dtype), tf.cast(llr_out, self._output_dtype)



class PolarDSCFlipDecoder(tf.keras.Model):
    """Hash/CRC-aided DSCF, order-1, with per-word frozen helper data.

    Keeps the user's SC decoder and flip-ranking idea; fixes candidate selection,
    output ordering and end-to-end key validation. For PUF calls, pass
    `frozen_values=[B,N]` and `expected_tags` from enrollment. Default verifier
    is KeyChecker('hash'). For legacy conventional CRC, crc_decoder may be used.

    The original implementation did not support omega > 1; this class explicitly
    declines multi-flip rather than silently pretending it is implemented.
    This implementation runs eagerly so Python hashlib can check candidate keys.
    """
    def __init__(self, frozen_pos, n, omga=1, crc_decoder=None, *,
                 checker=None, alpha=0.3, num_trials=2,
                 output_dtype=tf.float32, use_fast_sc=False,
                 use_flip_sc=True, **kwargs):
        super().__init__(dtype=output_dtype, **kwargs)
        self.n = int(n)
        self.omga = int(omga)
        if self.omga != 1:
            raise NotImplementedError('DSCF omega > 1 is not implemented in the supplied source.')
        self.alpha = float(alpha)
        self.num_trials = int(num_trials)
        if self.num_trials < 0:
            raise ValueError('num_trials must be nonnegative.')
        if checker is None and crc_decoder is None:
            raise ValueError('Provide checker (hash/CRC) or legacy crc_decoder.')
        self._checker = checker
        self._crc_decoder = crc_decoder
        self.decoder = PolarSCDecoder(
            frozen_pos=frozen_pos, n=self.n, output_dtype=output_dtype,
            use_fast_sc=use_fast_sc, use_flip_sc=use_flip_sc)
        self._info_bits = np.setdiff1d(
            np.arange(self.n), np.asarray(frozen_pos, dtype=np.int32)).astype(np.int32)
        self._output_dtype = output_dtype

    def _check(self, candidates, expected_tags):
        if self._checker is not None:
            if expected_tags is None:
                raise ValueError('PUF DSCF requires stored expected_tags from enrollment.')
            return self._checker.verify(candidates, expected_tags)
        _, valid = self._crc_decoder(tf.convert_to_tensor(candidates, dtype=tf.float32))
        return np.asarray(valid.numpy()).reshape(-1).astype(bool)

    @tf.function
    def _init_metric(self, llr_leaf_full):
        """Log of user's SC-Flip metric (avoids cumulative-product underflow)."""
        v = tf.gather(tf.abs(llr_leaf_full), self._info_bits, axis=1)
        a = tf.cast(self.alpha, tf.float32)
        # log (exp(-alpha |L_i|) prod_{j<=i} sigmoid(alpha |L_j|))
        return -a*v - tf.cumsum(tf.math.softplus(-a*v), axis=1)

    def call(self, x, frozen_values=None, expected_tags=None):
        if not tf.executing_eagerly():
            raise RuntimeError('Hash/CRC DSCF requires eager execution. Do not wrap in tf.function.')
        llr_ch = tf.reshape(tf.cast(x, tf.float32), [-1, self.n])
        fv = None if frozen_values is None else tf.cast(frozen_values, tf.float32)
        if fv is not None:
            fv = tf.reshape(fv, [-1, self.n])
        tags = (expected_tags.numpy() if tf.is_tensor(expected_tags)
                else expected_tags)

        u0, L0 = self.decoder(llr_ch, frozen_values=fv)
        out_u = np.asarray(u0.numpy(), dtype=np.float32).copy()
        out_L = np.asarray(L0.numpy(), dtype=np.float32).copy()
        first_keys = out_u[:, self._info_bits]
        passed = self._check(first_keys, tags)
        failed = np.flatnonzero(~passed)
        if self.num_trials == 0 or failed.size == 0:
            return tf.convert_to_tensor(out_u, dtype=self._output_dtype), tf.convert_to_tensor(out_L, dtype=self._output_dtype)

        metrics = self._init_metric(L0).numpy()
        rank = np.argsort(-metrics, axis=1, kind='stable')
        flips = self._info_bits[rank]
        for t in range(min(self.num_trials, self._info_bits.size)):
            if failed.size == 0:
                break
            positions = flips[failed, t]
            mask = tf.one_hot(positions, depth=self.n, dtype=tf.float32)
            sub_fv = None if fv is None else tf.gather(fv, failed, axis=0)
            u_try, L_try = self.decoder(tf.gather(llr_ch, failed, axis=0),
                                         flip_bits=mask, frozen_values=sub_fv)
            u_np = u_try.numpy()
            good = self._check(u_np[:, self._info_bits],
                               None if tags is None else tags[failed])
            if np.any(good):
                out_u[failed[good]] = u_np[good]
                out_L[failed[good]] = L_try.numpy()[good]
            failed = failed[~good]
        # If no flip passes, preserve the original SC estimate.
        return (tf.convert_to_tensor(out_u, dtype=self._output_dtype),
                tf.convert_to_tensor(out_L, dtype=self._output_dtype))
