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
        
if __name__ == '__main__':
    unittest.main()
