"""Run `python -m pytest -q test_puf.py` in a TensorFlow environment."""
import numpy as np
import pytest
pytest.importorskip('tensorflow')
import tensorflow as tf
from utils import polar_transform_bits, sample_puf_bits, sample_channel
from constructions import bhattacharyya_design
from Enrolment import SRAMPUFEnrollment
from keycheck import KeyChecker
from DSCFlip import PolarSCDecoder, PolarDSCFlipDecoder


def test_transform_is_self_inverse():
    rng = np.random.default_rng(123)
    x = sample_puf_bits(128, 16, rng=rng)
    assert np.array_equal(polar_transform_bits(polar_transform_bits(x)), x)


@pytest.mark.parametrize('channel', ['BSC', 'BEC', 'AWGN'])
@pytest.mark.parametrize('biased', [False, True])
def test_noiseless_recovery_with_nonzero_helper(channel, biased):
    n, k = 64, 8
    frozen, info, _ = bhattacharyya_design(n, k, channel='BSC', design_p=0.15)
    tagger = KeyChecker('hash')
    enrol = SRAMPUFEnrollment(n, frozen, biased=biased, prob_one=0.4, checker=tagger)
    rng = np.random.default_rng(123)
    record = enrol.enroll(12, rng=rng)
    # noiseless BEC p=0, BSC p=0, very high AWGN SNR
    y, L = sample_channel(record.x, channel=channel, p=0.0,
                           erasure_probability=0.0, snr_db=30.0,
                           prob_one=enrol.prob_one, rng=rng)
    dec = PolarSCDecoder(frozen, n)
    u_hat, _ = dec(tf.constant(L), frozen_values=tf.constant(record.frozen_full, tf.float32))
    key_hat = u_hat.numpy()[:, info].astype(np.uint8)
    assert np.array_equal(key_hat, record.key)
    assert np.all(tagger.verify(key_hat, record.tag))


def test_crc_and_hash_tags():
    rng = np.random.default_rng(17)
    keys = rng.integers(0, 2, (3, 128), dtype=np.uint8)
    for mode in ('hash','crc'):
        c = KeyChecker(mode)
        tag = c.tag(keys)
        assert c.verify(keys, tag).all()
        broken = keys.copy()
        broken[0, 2] ^= 1
        assert not c.verify(broken, tag)[0]


def test_dscf_keeps_batch_order_and_values_when_sc_is_correct():
    n, k = 32, 4
    frozen, info, _ = bhattacharyya_design(n, k)
    tagger = KeyChecker('hash')
    enrol = SRAMPUFEnrollment(n, frozen, checker=tagger)
    record = enrol.enroll(6, rng=np.random.default_rng(101))
    _, llr = sample_channel(record.x, p=0.0, rng=np.random.default_rng(42))
    dec = PolarDSCFlipDecoder(frozen, n, omga=1, checker=tagger, num_trials=2)
    u_hat, _ = dec.call(tf.constant(llr), frozen_values=tf.constant(record.frozen_full, tf.float32),
                        expected_tags=record.tag)
    assert np.array_equal(u_hat.numpy()[:, info].astype(np.uint8), record.key)
