"""Public key confirmation: SHA-256 digest or CRC, stored outside polar bits.

A PUF hash/CRC does not add redundant information bits to the 128-bit key.
It is used by DSCF to check candidate keys against the enrollment reference.
The public tag leaks information about the key, unlike the 2017 paper's
zero-leakage guarantee for W under uniform independent PUF bits.
"""
from __future__ import annotations
import hashlib
import numpy as np

# Same polynomial definitions as supplied crc.py.
_CRC_POLYS = {
    'CRC6': [6, 5, 0],
    'CRC8': [8, 4, 3, 1, 0],
    'CRC11': [11, 10, 9, 5, 0],
    'CRC16': [16, 12, 5, 0],
    'CRC24A': [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0],
    'CRC24B': [24, 23, 6, 5, 1, 0],
    'CRC24C': [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0],
}


def _batch_binary(keys):
    k = np.asarray(keys)
    if k.ndim == 1: k = k[None, :]
    if k.ndim != 2 or not np.all((k == 0) | (k == 1)):
        raise ValueError('keys must be binary [B,K] or [K].')
    return k.astype(np.uint8)


class KeyChecker:
    def __init__(self, method: str = 'hash', hash_bits: int = 128,
                 crc_degree: str = 'CRC16', hash_name: str = 'sha256'):
        self.method = method.lower()


        if self.method not in ('hash', 'crc', 'none'):
            raise ValueError('method: hash, crc, or none')

        
        if hash_bits not in (64, 96, 128, 160, 192, 224, 256):
            raise ValueError('hash_bits must be a supported byte-aligned SHA-256 truncation.')

        
        if crc_degree not in _CRC_POLYS:
            raise ValueError(f'Unknown CRC degree {crc_degree}')
        
        if hash_name not in ('sha256', 'sha3_256', 'sha512', 'blake2s'):
            raise ValueError('hash_name: sha256, sha3_256, sha512, or blake2s.')

        
        self.hash_name = hash_name
        self.hash_bits = hash_bits
        self.crc_degree = crc_degree
        self._crc_matrix_cache = {}

    def _crc_mat(self, k):
        if k not in self._crc_matrix_cache:
            degree = max(_CRC_POLYS[self.crc_degree])
            poly = np.zeros(degree + 1, dtype=np.uint8)
            for i in _CRC_POLYS[self.crc_degree]: poly[degree - i] = 1
            mat = np.zeros((k, degree), dtype=np.uint8)
            # The original CRC encoder's generator matrix algorithm.
            rem = np.zeros(degree, dtype=np.uint8)
            rem[0] = 1
            for i in range(k):
                extended = np.concatenate((rem, np.zeros(1, dtype=np.uint8)))
                if extended[0]: extended ^= poly
                rem = extended[1:]
                mat[k-i-1] = rem
            self._crc_matrix_cache[k] = mat
        return self._crc_matrix_cache[k]

    def tag(self, keys: np.ndarray) -> np.ndarray:
        k = _batch_binary(keys)
        if self.method == 'none':
            return np.empty((k.shape[0],0), dtype=np.uint8)
        if self.method == 'crc':
            return (k.astype(np.int64) @ self._crc_mat(k.shape[1]).astype(np.int64) % 2).astype(np.uint8)
        out = np.zeros((k.shape[0], self.hash_bits//8), dtype=np.uint8)
        # Domain separation and exact bit length (avoid padded-bit ambiguity).
        header = b'PUF-KEY-CHECK-v1:' + k.shape[1].to_bytes(4, 'big')
        for i, row in enumerate(k):
            digest = hashlib.new(self.hash_name, header + np.packbits(row).tobytes()).digest()
            out[i] = np.frombuffer(digest[:self.hash_bits//8], dtype=np.uint8)
        return out

    def verify(self, candidate_keys: np.ndarray, expected_tags: np.ndarray) -> np.ndarray:
        keys = _batch_binary(candidate_keys)
        tags = np.asarray(expected_tags, dtype=np.uint8)
        if tags.ndim == 1: tags = tags[None,:]
        got = self.tag(keys)
        if tags.shape != got.shape:
            raise ValueError(f'Expected tag shape {got.shape}, received {tags.shape}.')
        return np.all(got == tags, axis=1)
