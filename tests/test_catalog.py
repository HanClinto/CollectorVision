import unittest

import numpy as np

from collector_vision.catalog import pack_ids


class PackIdsTests(unittest.TestCase):
    def test_pack_ids_supports_empty_string_as_zero_row(self) -> None:
        packed = pack_ids([""])
        self.assertEqual(packed.shape, (1, 16))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertTrue(np.array_equal(packed[0], np.zeros(16, dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
