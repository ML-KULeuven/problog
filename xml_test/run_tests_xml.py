import unittest

from xmlrunner import XMLTestRunner

loader = unittest.TestLoader()
tests = loader.discover(".")
runner = XMLTestRunner(output='test-reports')
runner.run(tests)
