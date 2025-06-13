import unittest

from .test_all import TestRustPuto

if __name__ == "__main__":
    suite = unittest.TestSuite([TestRustPuto()])
    unittest.main()
