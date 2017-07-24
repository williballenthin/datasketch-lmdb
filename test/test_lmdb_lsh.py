import shutil
import unittest
import contextlib
from hashlib import sha1

import numpy as np

from datasketch_lmdb import LMDBMinHashLSH
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator


class TestLMDBMinHashLSH(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree('lsh-test.bin')

    def test_init(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.8)) as lsh:
            self.assertTrue(lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.8, weights=(0.2,0.8))) as lsh:
            b2, r2 = lsh.b, lsh.r
            self.assertTrue(b1 < b2)
            self.assertTrue(r1 > r2)

    def test_insert(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=16)) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=16)) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            m3 = MinHash(18)
            self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=16)) as lsh:
            m1 = MinHash(16)
            m1.update("a".encode("utf8"))
            m2 = MinHash(16)
            m2.update("b".encode("utf8"))
            lsh.insert("a", m1)
            lsh.insert("b", m2)

            lsh.remove("a")

            self.assertRaises(ValueError, lsh.remove, "c")


class TestWeightedLMDBMinHashLSH(unittest.TestCase):

    def tearDown(self):
        shutil.rmtree('lsh-test.bin')

    def test_init(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.8)) as lsh:
            self.assertTrue(lsh.is_empty())
            b1, r1 = lsh.b, lsh.r

        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.8, weights=(0.2,0.8))) as lsh:
            b2, r2 = lsh.b, lsh.r
            self.assertTrue(b1 < b2)
            self.assertTrue(r1 > r2)

    def test_insert(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=4)) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            self.assertTrue("a" in lsh)
            self.assertTrue("b" in lsh)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            self.assertRaises(ValueError, lsh.insert, "c", m3)

    def test_query(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=4)) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            lsh.insert("a", m1)
            lsh.insert("b", m2)
            result = lsh.query(m1)
            self.assertTrue("a" in result)
            result = lsh.query(m2)
            self.assertTrue("b" in result)

            mg = WeightedMinHashGenerator(10, 5)
            m3 = mg.minhash(np.random.uniform(1, 10, 10))
            self.assertRaises(ValueError, lsh.query, m3)

    def test_remove(self):
        with contextlib.closing(LMDBMinHashLSH(path='lsh-test.bin', threshold=0.5, num_perm=4)) as lsh:
            mg = WeightedMinHashGenerator(10, 4)
            m1 = mg.minhash(np.random.uniform(1, 10, 10))
            m2 = mg.minhash(np.random.uniform(1, 10, 10))
            lsh.insert("a", m1)
            lsh.insert("b", m2)

            lsh.remove("a")

            self.assertRaises(ValueError, lsh.remove, "c")


if __name__ == "__main__":
    unittest.main()
