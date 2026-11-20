import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pca_py  # noqa: E402


class EigenDecomposeTest(unittest.TestCase):
    def test_diagonal_matrix(self):
        vals, vecs = pca_py.eigen_decompose([[1.0, 0, 0], [0, 5.0, 0], [0, 0, 3.0]])
        for got, want in zip(vals, [5.0, 3.0, 1.0]):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(abs(vecs[0][1]), 1.0, places=4)

    def test_known_symmetric_matrix(self):
        # [[2,1,0],[1,2,0],[0,0,1]] has eigenvalues 3, 1, 1
        vals, _ = pca_py.eigen_decompose([[2.0, 1, 0], [1, 2.0, 0], [0, 0, 1.0]])
        for got, want in zip(vals, [3.0, 1.0, 1.0]):
            self.assertAlmostEqual(got, want, places=4)

    def test_eigenvectors_are_orthonormal(self):
        _, vecs = pca_py.eigen_decompose([[4.0, 2, 0.6], [2, 3.0, 0.4], [0.6, 0.4, 1.0]])
        for i, v in enumerate(vecs):
            self.assertAlmostEqual(pca_py.dot(v, v), 1.0, places=4)
            for w in vecs[i + 1:]:
                self.assertAlmostEqual(pca_py.dot(v, w), 0.0, places=4)


class ProcessImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "in.png")
        self.dst = os.path.join(self.tmp.name, "out.png")
        img = Image.new("RGB", (8, 8))
        img.putdata([((x * 31) % 256, (y * 17) % 256, ((x + y) * 13) % 256)
                     for y in range(8) for x in range(8)])
        img.save(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_components_reconstructs_original(self):
        pca_py.process_image(self.src, self.dst, k=3)
        original = list(Image.open(self.src).getdata())
        result = list(Image.open(self.dst).getdata())
        for a, b in zip(original, result):
            for ca, cb in zip(a, b):
                self.assertEqual(ca, cb)

    def test_greyscale_image_all_components_is_lossless(self):
        # R = G = B gives a rank-1 covariance with two zero eigenvalues
        img = Image.new("RGB", (8, 8))
        img.putdata([((x * 37 + y * 11) % 256,) * 3 for y in range(8) for x in range(8)])
        img.save(self.src)
        pca_py.process_image(self.src, self.dst, k=3)
        original = list(Image.open(self.src).getdata())
        result = list(Image.open(self.dst).getdata())
        for a, b in zip(original, result):
            for ca, cb in zip(a, b):
                self.assertEqual(ca, cb)

    def test_rejects_invalid_k(self):
        for k in (0, 4):
            with self.assertRaises(ValueError):
                pca_py.process_image(self.src, self.dst, k=k)


if __name__ == "__main__":
    unittest.main()
