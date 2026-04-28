import unittest

import numpy as np

from collector_vision.interfaces import DetectionResult


class DetectionResultTests(unittest.TestCase):
    def test_dewarp_outputs_embedder_sized_square_crop(self) -> None:
        bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        crop = DetectionResult(corners=corners, card_present=True).dewarp(bgr)

        self.assertEqual(crop.size, (448, 448))


if __name__ == "__main__":
    unittest.main()
