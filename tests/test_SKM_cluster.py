from unittest import TestCase, skipIf, skipUnless
import numpy as np
from numpy.testing import (assert_raises, assert_equal, assert_almost_equal)
from os.path import (exists, join, split, abspath)
import os
from .. import SKM_cluster as skm




class Test_SKM_Cluster(TestCase):

    def setUp(self):
        """ create small example data """
        prng = np.random.RandomState(42)
        rand_dat = prng.random_sample((10,20))
        ids = ['B%03d'%x for x in range(len(rand_dat))]
        features = ['feature_%04d'%x for x in range(rand_dat.shape[1])]
        df = skm.pd.DataFrame(rand_dat, index = ids, columns=features)
        self.data = df
        self.homedir = os.environ['HOME']
        data = np.zeros((60,100))
        data[:10] = .5
        data[10:10+20] = .2
        data[30:] = .6
        clust_dat = 10* [1] + 20 * [2] + 30 * [3]
        ids = ['B%03d'%x for x in range(len(data))]
        features = ['feature_%04d'%x for x in range(data.shape[1])]
        data_df = skm.pd.DataFrame(data, index=ids, columns=features)
        clust_df = skm.pd.DataFrame({'cluster':clust_dat}, index=ids)
        self.data_df = data_df
        self.clust_df = clust_df
        self.features = features


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

    def test_create_tightclust(self):
        tmp = np.round(self.data.values).astype(str)
        tmp[:4,:] = '0'
        tmp[6:,:] = '1'
        tmpdf = skm.pd.DataFrame(tmp)
        # raise error of data not 'pos' and 'neg' values
        assert_raises(KeyError, skm.create_tightclust, tmpdf)
        # create a dataframe with 'pos' and 'neg' values
        membership_data = skm.pd.DataFrame(tmp)
        membership_data[membership_data == '0'] = 'neg'
        membership_data[membership_data == '1'] = 'pos'
        tight_subjects = skm.create_tightclust(membership_data)
        assert_equal(tight_subjects.index.values, np.array([0,1,2,3,6,7,8,9]))

    def test_skm_permute(self):
        tmp = self.data.values.copy()
        tmp = (tmp - 0.5) / 2.0
        tmp[:5, :10] = tmp[:5, :10] - 0.9
        tmp[5:, :10] = tmp[5:, :10] + 0.9
        tmpdf = skm.pd.DataFrame(tmp)
        best_L1bound, lowest_L1bound = skm.skm_permute(tmpdf)
        assert_equal(best_L1bound> lowest_L1bound, True)
        assert_almost_equal(best_L1bound, 3.518520979)

    def test_skm_cluster(self):
        tmp = self.data.values.copy()
        tmp = (tmp - 0.5) / 2.0
        tmp[:5, :10] = tmp[:5, :10] - 0.9
        tmp[5:, :10] = tmp[5:, :10] + 0.9
        tmpdf = skm.pd.DataFrame(tmp)
        best_L1bound = 3.518520979
        km_weight, km_clust = skm.skm_cluster(tmpdf, best_L1bound)
        assert_equal(km_clust[0].values, np.array(5*[1] + 5 * [2]))

    def test_elbow(self):
        big = np.linspace(1,.25,50)
        little = np.linspace(.25,0,50)
        all = np.concatenate((big, little))
        all = all / all.sum()
        index = ['reg_%04d'%x for x in range(all.shape[0])]
        alldf = skm.pd.DataFrame(all, index = index, columns=('weights',))
        topfeat, sweights, elbow = skm.find_elbow(alldf)
        assert_equal(elbow, 48)


    def test_parse_clusters(self):
        # define 3 groups
        top_features = self.features[:10]
        cluster_dats = skm.parse_clusters(self.clust_df, self.data_df, 
                top_features)
        assert_equal(cluster_dats[0].shape, (10,10))
        assert_equal(cluster_dats[1].shape, (20,10))
        assert_equal(cluster_dats[2].shape, (30,10))


    def test_parse_pos_neg(self):
        assert_raises(ValueError, skm.parse_pos_neg, [100,100,100], [1,2,3])

    def test_make_vector(self):
        p0 = [0, 2]
        px = [100,0]
        vect = skm.make_vector(p0,px)
        assert_equal(vect, np.array([100,-2]))

    def test_predict_clust(self):
        sample_data = self.data
        cutoffs = skm.pd.Series(20*[.95], index = sample_data.columns)
        predicted = skm.predict_clust(sample_data, cutoffs)
        expected = np.array(['pos', 'pos', 'pos', 'pos', 'neg', 'neg',
            'pos', 'pos', 'neg', 'neg'], dtype=str)
        assert_equal(predicted.values, expected)
        # test error if DataFrame is sent
        cutoffs_df = skm.pd.DataFrame(20*[.95], 
                index = sample_data.columns)
        assert_raises(TypeError, skm.predict_clust, sample_data, cutoffs_df)


if __name__ == '__main__':
    unittest.main()
