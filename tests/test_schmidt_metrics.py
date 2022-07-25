import unittest
import pytest
import numpy as np
import numpy.testing as npt

from utility_functions.metrics import get_segments, get_centers, get_schmidt_tp_fp, schmidt_metrics


@pytest.mark.tm
class TestCanonicalSimplex(unittest.TestCase):
    def setUp(self):
        pass

    def test_segments(self):
        """
        Check if segment boundaries are correct
        """
        # Repeating events 0->1->2->3->1
        test1 = np.array([0] * 60 +
                         [1] * 150 +
                         [2] * 70 +
                         [3] * 400 +
                         [0] * 120)
        expected = np.array([[0, 59, 0],
                             [60, 209, 1],
                             [210, 279, 2],
                             [280, 679, 3],
                             [680, 799, 0]])
        npt.assert_array_equal(expected, get_segments(test1))

        # 1 observation event beginning and end
        test2 = np.array([0] +
                         [1] * 120
                         + [2])
        expected = np.array([[0, 0, 0],
                             [1, 120, 1],
                             [121, 121, 2]])
        npt.assert_array_equal(expected, get_segments(test2))

        # 1 observation event middle
        test3 = np.array([0] * 60 +
                         [2] +
                         [3] * 60)
        expected = np.array([[0, 59, 0],
                             [60, 60, 2],
                             [61, 120, 3]])
        npt.assert_array_equal(expected, get_segments(test3))

        # 1 observation event end
        test4 = np.array([0] * 60 +
                         [3] * 60 +
                         [0])
        expected = np.array([[0, 59, 0],
                             [60, 119, 3],
                             [120, 120, 0]])
        npt.assert_array_equal(expected, get_segments(test4))
