"""
Module name
"""

from __future__ import print_function

from problog import root_path
from problog.util import subprocess_call, subprocess_check_output
import unittest
import os
import sys
import glob

if __name__ == '__main__' :
    sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestLFI(unittest.TestCase) :
    def setUp(self):
        try:
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError:
            self.assertSequenceEqual = self.assertCountEqual

def read_result(filename) :
    results = []
    with open( filename ) as f :
        reading = False
        for l in f :
            l = l.strip()
            if l.startswith('%Expected outcome:') :
                reading = True
            elif reading :
                if l.lower().startswith('% error') :
                    return l[len('% error'):].strip()
                elif l.startswith('% ') :
                    res = l[2:]
                    results.append(res)
                else :
                    reading = False
    return results

def createTestLFI(filename, useparents = False):
    def test(self) :
        problogcli = root_path('problog-cli.py')

        model = filename
        examples = filename.replace(".pl", ".ev")
        out_model = filename.replace(".pl", ".l_pl")

        expected = read_result(model)

        if not os.path.exists(examples):
            raise Exception("Evidence file is missing: {}".format(examples))
        if useparents:
            out = subprocess_check_output([sys.executable, problogcli, 'lfi', '-n', '10', '-O',
                                       model.replace('.pl', '.l_pl'), model, examples, '--useparents'])



        with open(out_model, "r") as f:
            outlines = f.readlines()
        outlines = [line.strip() for line in outlines]
        assert(expected == outlines)
    return test


if __name__ == '__main__':
    filenames = sys.argv[1:]
else:
    ADfilenames = glob.glob(root_path('test', 'lfi', 'AD_positive', '*.pl'))
    simple_filenames = glob.glob(root_path('test', 'lfi', 'simple', '*.pl'))

# tests for simple cases (non-ADs)
for testfile in simple_filenames :
    testname = 'test_lfi_simple_' + os.path.splitext(os.path.basename(testfile))[0]
    setattr( TestLFI, testname, createTestLFI(testfile, True))

# tests for ADs
for testfile in ADfilenames :
    testname = 'test_lfi_AD_' + os.path.splitext(os.path.basename(testfile))[0]
    setattr( TestLFI, testname, createTestLFI(testfile, True))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLFI)
    unittest.TextTestRunner(verbosity=2).run(suite)
