import unittest

import numpy as np
from fastapi import HTTPException

from scripts.app.services.model import FEATURE_NAMES, N_FEATURES, vectorize


class TestVectorize(unittest.TestCase):
    def test_list_of_correct_length(self):
        arr = vectorize([0.0] * N_FEATURES)
        self.assertEqual(arr.shape, (N_FEATURES,))

    def test_wrong_length_raises(self):
        with self.assertRaises(HTTPException):
            vectorize([0.0] * 5)

    def test_dict_is_reordered(self):
        payload = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
        arr = vectorize(payload)
        self.assertEqual(list(arr), [float(i) for i in range(N_FEATURES)])

    def test_missing_feature_raises(self):
        payload = {name: 0.0 for name in FEATURE_NAMES[:-1]}
        with self.assertRaises(HTTPException):
            vectorize(payload)


if __name__ == "__main__":
    unittest.main()
