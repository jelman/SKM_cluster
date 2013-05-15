from unittest import TestCase, skipIf, skipUnless
import numpy as np
from numpy.testing import (assert_raises, assert_equal, assert_almost_equal)
from os.path import (exists, join, split, abspath)
import os
from .. import SKM_cluster as skm




class Test_SKM_Cluster(TestCase):

    def setUp(self):
        """ create small example data """
        pass

    def test_run_command(self):
        return_code = skm.run_command('echo')
        assert_equal(return_code, 0)
        # test failure for missing function
        return_code = skm.run_command('blueberry')
        assert_equal(return_code, 127)

    def test_get_sparcl_dir(self):
        homedir = os.environ['HOME']
        sparcl_dir = skm.get_sparcl_dir()
        assert_equal(homedir in sparcl_dir, True)
        assert_equal('rpackages' in sparcl_dir, True)
        assert_equal('sparcl' in sparcl_dir, True)



if __name__ == '__main__':
    unittest.main()
