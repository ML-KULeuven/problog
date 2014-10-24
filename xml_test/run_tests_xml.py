import unittest

from xmlrunner import XMLTestRunner

loader = unittest.TestLoader()
tests = loader.discover(".")
runner = XMLTestRunner()
runner.run(tests)
