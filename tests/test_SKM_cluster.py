from unittest import TestCase, skipIf, skipUnless
import numpy as np
from numpy.testing import (assert_raises, assert_equal, assert_almost_equal)
from os.path import (exists, join, split, abspath)
import os
from .. import SKM_cluster as skm




class Test_SKM_Cluster(TestCase):

    def setUp(self):
        """ create small example data """
        rand_dat = np.random.random((10,20))
        df = skm.pd.DataFrame(rand_dat)
        self.data = df
        self.homedir = os.environ['HOME']

    def test_run_command(self):
        return_code = skm.run_command('echo')
        assert_equal(return_code, 0)
        # test failure for missing function
        return_code = skm.run_command('blueberry')
        assert_equal(return_code, 127)

    def test_get_sparcl_dir(self):
        homedir = self.homedir
        sparcl_dir = skm.get_sparcl_dir()
        assert_equal(homedir in sparcl_dir, True)
        assert_equal('rpackages' in sparcl_dir, True)
        assert_equal('sparcl' in sparcl_dir, True)

    def test_import_sparcl(self):
        ## test that sparcl is installed
        sparcl = skm.import_sparcl()
        assert_equal(skm.sparcl_installed(), True)


    def test_sample_data(self):
        # test default 70% split
        train, test = skm.sample_data(self.data)
        assert_equal(train.shape[0], 7)
        assert_equal(test.shape[0], 3)
        # test 50% split
        train, test = skm.sample_data(self.data, split = .5)
        assert_equal(train.shape[0], 5)
        assert_equal(test.shape[0], 5)


if __name__ == '__main__':
    unittest.main()
