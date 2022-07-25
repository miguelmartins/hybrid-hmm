import unittest
import pytest
import numpy as np
import numpy.testing as npt

from utility_functions.metrics import get_segments, get_centers, get_schmidt_tp_fp, schmidt_metrics


@pytest.mark.tm
class TestSchmidtMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_segments(self):
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

    def test_get_centers(self):
        """
        Test interface to show get center of windows
        """

        # Single state duration
        test1 = np.array([0] +
                         [1] * 3)
        expected = np.array([[0.0, 0],
                             [2., 1]])
        segments = get_segments(test1)
        np.testing.assert_array_equal(expected, get_centers(segments))

        # Trivial case all even
        test2 = np.array([0] * 2 +
                         [1] * 2 +
                         [2] * 2)
        segments = get_segments(test2)
        expected = np.array([[0.5, 0],
                             [2.5, 1],
                             [4.5, 2]])
        np.testing.assert_array_equal(expected, get_centers(segments))

        # Single state beginning + trivial case even
        test3 = np.array([3] +
                         [0] * 2 +
                         [1] * 2 +
                         [2] * 2)
        segments = get_segments(test3)
        expected = np.array([[0, 3],
                             [1.5, 0],
                             [3.5, 1],
                             [5.5, 2]])
        np.testing.assert_array_equal(expected, get_centers(segments))

        # Single state middle + trivial case
        test4 = np.array([0] * 2 +
                         [1] * 2 +
                         [3] +
                         [2] * 2)
        segments = get_segments(test4)
        expected = np.array([[0.5, 0],
                             [2.5, 1],
                             [4, 3],
                             [5.5, 2]])
        np.testing.assert_array_equal(expected, get_centers(segments))

        # Odd durations
        test5 = np.array([0] * 3 +
                         [1] * 3 +
                         [2] * 3)
        segments = get_segments(test5)
        expected = np.array([[1, 0],
                             [4, 1],
                             [7., 2]])
        np.testing.assert_array_equal(expected, get_centers(segments))

    def test_schmidt_tp_fp(self):
        """
        Test tp, fp and total calculation for a set of examples and possible corner cases
        """

        # Perfect S+ and PPV for simple sound
        y_true = np.array([0] * 100 +
                           [2] * 100)
        y_pred = y_true
        exp_tp, exp_fp, exp_total = 2, 0, 2
        tp, fp, total = get_schmidt_tp_fp(y_true, y_pred)
        assert (exp_tp == tp) and (exp_fp == fp) and (exp_total == total)

        # Worst case S+ and PPV for simple sound
        y_true = np.array([0] * 100 +
                          [2] * 100)
        y_pred = np.array([2] * 100 +
                          [0] * 100)
        exp_tp, exp_fp, exp_total = 0, 2, 2
        tp, fp, total = get_schmidt_tp_fp(y_true, y_pred)
        assert (exp_tp == tp) and (exp_fp == fp) and (exp_total == total)