import unittest
import sys, os

# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from core.costs import full_affinity_v2, squared_distance_v2
import numpy as np
import tensorflow as tf
from keras import backend as K

class TestFullAffinity(unittest.TestCase):

    def test_squared_distance_v2(self):
        X = np.array(
            [
                [0, 0],
                [3, 4],
                [1, 1]
            ]
        )

        D = np.array(
            [
                [0, 25, 2],
                [25, 0, 13],
                [2, 13, 0]
            ]
        )

        self.assertTrue(np.array_equal(squared_distance_v2(X), D))        

    def test_full_affinity_v2(self):
        X = np.array(
            [
                [0, 0],
                [3, 4],
                [1, 1]
            ]
        )

        n_nbrs = 3

        A = np.diag([.2, .2, 1.0/np.sqrt(13)])
        D = np.array(
            [
                [0, 25, 2],
                [25, 0, 13],
                [2, 13, 0]
            ]
        )

        res1 = np.exp(-A@D@A)
        res2 = full_affinity_v2(X, n_nbrs)

        # self.assertEqual(np.shape(res1), np.shape(res2))

        # for i in range(np.shape(res1)[0]):
        #     for j in range(np.shape(res1)[1]):
        #         self.assertAlmostEqual(res1[i, j], res2[i, j])
        self.assertTrue(np.allclose(res1, res2))

    # def test_sum_tuple(self):
    #     self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
