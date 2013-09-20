import os, sys
import argparse

import numpy as np
import pandas as pd
import pandas.rpy.common as com
import rpy2
import random
import subprocess


def r_installed():
    """ check that r is installed on system
    does not work on Windows """
    try:
        subprocess.check_output(['which','R'])
        return True
    except:
        if 'win' in sys.platform:
            print 'Warning: unable to check for R in windows'
        return False

 
def run_command(cmd):
    """ run command, return stdout, stderr, returncode"""
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Child returned", retcode
        return retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        return None

def get_sparcl_dir():
    """ returns default location of sparcl library dir for this user"""
    home = os.environ['HOME']
    return os.path.join(home, '.rpackages', 'sparcl')
    

def sparcl_installed():
    """ check if sparcl library is installed in default location
    (default: /userhome/.rpackages/sparcl) """   
    return os.path.isdir(get_sparcl_dir())

def install_sparcl():
    home = os.environ['HOME']
    rpkg_dir = os.path.join(home, '.rpackages')
    if not os.path.isdir(rpkg_dir):
        os.mkdir(rpkg_dir)
    curr_dir, _ = os.path.split(__file__)
    sparcl_loc = os.path.join(curr_dir, 'sparcl')
    cmd = ' '.join(['R', 'CMD', 'INSTALL', sparcl_loc, '-l', rpkg_dir])
    run_command(cmd)
    

def import_sparcl():
    if not sparcl_installed():
        install_sparcl()
    sparcl_dir = get_sparcl_dir()
    rpkg_dir, _ = os.path.split(sparcl_dir)
    sparcl = rpy2.robjects.packages.importr("sparcl", lib_loc=rpkg_dir)
    return sparcl


def create_rslts_frame(dataframe):  
    """ Create empty pandas dataframe to hold feature weights and cluster 
    membership results of each resampling run""" 
    weight_rslts = pd.DataFrame(data=None, index = dataframe.columns)
    clust_rslts = pd.DataFrame(data=None, index = dataframe.index)
    return weight_rslts, clust_rslts
    
    
def sample_data(data, split = .7):
    """
    Takes an array of data as input. Randomly samples 70% of the observations
    and returns as an array
    """
    samp_n = int(split * len(data))
    rand_samp = sorted(random.sample(xrange(len(data)), samp_n))
    sampdata = data.take(rand_samp)
    unsamp_idx = [x for x in xrange(len(data)) if x not in rand_samp]
    unsampdata = data.take(unsamp_idx)
    return sampdata, unsampdata


def skm_permute(data):
    """
    rpy2 wrapper for R function: KMeansSparseCluster.permute from the 
    sparcl package.
    The tuning parameter controls the L1 bound on w, the feature weights. 
    A permutation approach is used to select the tuning parameter. 
    
    Infile
    ------
    data: pandas Dataframe
        n by p dataframe where n is observations and p is features (i.e. ROIs)
        Should be a pandas DataFrame with subject codes as index and features 
        as columns.
    
    Returns
    -------
    best_L1bound: float
    	tuning parameter that returns the highest gap statistic
	    (more features given non-zero weights)
    
    lowest_L1bound: float
    	smallest tuning parameter that gives a gap statistic within 
        one sdgap of the largest gap statistic (sparser result)
    """
    sparcl = import_sparcl()
    r_data = com.convert_to_r_dataframe(data)
    km_perm = sparcl.KMeansSparseCluster_permute(r_data,K=2,nperms=25)
    best_L1bound = km_perm.rx2('bestw')[0]
    wbounds = km_perm.rx2('wbounds')
    gaps = km_perm.rx2('gaps')
    bestgap = max(gaps)
    sdgaps = km_perm.rx2('sdgaps')
    # Calculate smallest wbound that returns gap stat 
    # within one sdgap of best wbound
    wbound_rnge = [wbounds[i] for i in range(len(gaps)) if \
            (gaps[i]+sdgaps[i]>=bestgap)]
    lowest_L1bound = min(wbound_rnge)
    return best_L1bound, lowest_L1bound
    
    
def skm_cluster(data, L1bound):
    """
    rpy2 wrapper for R function: KMeansSparseCluster from the sparcl package.
    This function performs sparse k-means clustering. 
    You must specify L1 bound on w, the feature weights. 
    
    Note: A smaller L1 bound will results in sparser weighting. 
    If a large number of features are included, it may be useful to use 
    the smaller tuning parameter returned by 
    KMeansSparseCluster.permute wrapper function.
    
    Infile
    ------
    data: pandas Dataframe
        n by p dataframe where n is observations and p is features (i.e. ROIs)
        Should be a pandas DataFrame with subject codes as index and features 
        as columns.
    
    Returns
    -------
    km_weights: pandas DataFrame   
        index = feature labels, values = weights
    km_clusters: pandas DataFrame
        index = subject code, values = cluster membership
    """
    sparcl = import_sparcl()
    # Convert pandas dataframe to R dataframe
    r_data = com.convert_to_r_dataframe(data)
    # Cluster observations using specified L1 bound
    km_out = sparcl.KMeansSparseCluster(r_data,K=2, wbounds=L1bound)
    # Create dictionary of feature weights, normalized feature weights and
    # cluster membership
    ws = km_out.rx2(1).rx2('ws')
    km_weights = {k.replace('.','-'): [ws.rx2(k)[0]] for k in ws.names}
    km_weights = pd.DataFrame.from_dict(km_weights)
    km_weightsT = km_weights.T
    km_weightsnorm = km_weightsT/km_weightsT.sum()
    Cs = km_out.rx2(1).rx2('Cs')
    km_clusters = {k: [Cs.rx2(k)[0]] for k in Cs.names}   
    km_clusters = pd.DataFrame.from_dict(km_clusters)
    km_clusters = km_clusters.T
    return km_weightsnorm, km_clusters
       

def make_vector(p0, px):
    """ given two points, create numpy vector array"""
    p0 = np.asarray(p0)
    px = np.asarray(px)
    return px - p0

def find_elbow(weights):
    """ given the weights, sort, and determin the curve `elbow` by
    finding the point that is furthest from the vector connecting
    the largest and smallest weight"""
    columns = weights.columns
    sorted_weights = weights.sort(columns=columns[0], ascending=False)
    sweights = sorted_weights[columns[0]].values
    first = np.array([0,sweights[0]])
    # b is vector between largest and smallest weight
    b = make_vector(first,[sweights.shape[0], sweights[-1]])
    # normalize vector
    bhat = b / np.sqrt((b**2).sum())
    # all but first weight
    spoints = np.array([[val+1,x] for val, x in enumerate(sweights[1:])])
    firstbig = np.tile(first.T, sweights.shape[0]-1)
    firstbig.shape = spoints.shape
    # vectors between largest weight and subsequent weight(s)
    curve_vectors = spoints - firstbig
    bhat_big = np.tile(bhat, sweights.shape[0]-1)
    bhat_big.shape = spoints.shape
    # project vector onto normalized b vector
    tmp = map(np.dot, curve_vectors, bhat_big)
    tmp2 = map(np.multiply, tmp, bhat_big)
    # calc orth distance from b-vector to point, find largest dist
    elbow = ((curve_vectors - tmp2)**2).sum(axis=1).argmax()
    # find weights >= elbow weight
    columns = sorted_weights.columns
    mask = sorted_weights[columns[0]] >= sorted_weights[columns[0]][elbow]
    top_features = sorted_weights[mask].index
    ## return top features
    return top_features, sorted_weights, elbow





def calc_cutoffs(data, weights, weightsum, clusters):
    """
    Determines the cluster centers for regions. This will only detmine cutoffs 
    for features with weights making up the top percentile specified. Clusters 
    are then labelled as positive or negative based on which has the highest 
    average PIB index across these features. This is done because clustering 
    does not consistently assign the same label to groups across runs. 
    
    Inputs
    ------
    data:   pandas Dataframe
        n by p dataframe where n is observations and p is features (i.e. ROIs)
        Should be a pandas DataFrame with subject codes as index and features 
        as columns.
            
    weights:    pandas Dataframe   
        index = feature labels, values = weights
    weightsum:  float
        cluster cut-offs will be determined only for features with weights
        in the top percentile specified
    clusters:   pandas Dataframe
        index = subject code, values = cluster membership
    
    Returns
    -------
    clust_renamed:  pandas Dataframe
        renamed version of clusters input such that integer labels
        are replaced with appropriate string label ('pos' or 'neg')
    
    cutoffs:    pandas Dataframe
        index = features labels accounting for top 50% of weights
        values = cut-off score determined as mid-point of cluster means
        for each feature
    
    Note:
    The cluster means may be used to create cut-off scores (i.e. the midpoint 
    between cluster means) in order to predict cluster membership of other 
    subjects. If a large number of features were used, it may be useful to 
    set a lower weightsum (i.e., .50). This will constrain the features used
    to classify subjects to only those that contributed most to clustering. 
    """
    # Select features accounting for up to 50% of the weighting
    sorted_weights = weights.sort(columns=0, ascending=False)
    sum_weights = sorted_weights.cumsum()
    topfeats = sum_weights[sum_weights[0] <= weightsum].index
    clust1subs = clusters[clusters[0] == 1].index
    clust2subs = clusters[clusters[0] == 2].index
    clust1dat = data.reindex(index=clust1subs, columns=topfeats)
    clust2dat = data.reindex(index=clust2subs, columns=topfeats)
    clust1mean = clust1dat.mean(axis=0)
    clust2mean = clust2dat.mean(axis=0)

    ## XXXX NOTE 1, 2 are arbitrary, not based on cluster mean
    if clust1mean.mean() > clust2mean.mean():
        pos_means = clust1mean
        neg_means = clust2mean
        clust_renamed = clusters[0].astype(str).replace(['1','2'], 
                                                        ['pos','neg'])
    elif clust1mean.mean() < clust2mean.mean():
        pos_means = clust2mean
        neg_means = clust1mean
        clust_renamed = clusters[0].astype(str).replace(['1','2'], 
                                                        ['neg','pos'])
    cutoffs = (pos_means + neg_means) / 2
    clust_renamed.name = 'PIB_Status'
    return clust_renamed, cutoffs


def predict_clust(data, cutoffs):
    """
    Predict cluster membership for set of subjects using feature cutoff scores.
    Classifies as postive if any feature surpasses cut-off value.
    
    Inputs
    ------
    data:   pandas DataFrame  
        n by p dataframe where n is observations and p is features (i.e. ROIs)
        Should include subject codes as index and features as columns.
            
    cutoffs:   pandas DataFrame   
        index = features of interest labels, values = weights 
    
    Returns
    -------
    predicted_clust:    pandas DataFrame
        index = subject codes passed from input data
        values = predicted cluster membership     
    """
    # Check all features against cut-offs
    cutdata = data[cutoffs.index] > cutoffs 
    # Classify as pos if any feature is above cutoff
    cutdata_agg = cutdata.any(axis=1)
    predicted_clust = cutdata_agg.astype(str).replace(['T', 'F'], 
                                                      ['pos', 'neg'])
    predicted_clust.name = 'PIB_Status'
    return predicted_clust
    
    
def create_tightclust(clusterdata):
    """
    Creates tight clusters consisting of subjects classified as a member 
    of given cluster in at least 96% of resample runs.
    
    Inputs
    ------
    clusterdata: pandas Dataframe
        Dataframe containing cluster membership over all resamples
        index = subject code 
        columns = resample run 
        values = cluster membership
    
    Returns
    -------
    tight_subs: pandas Dataframe 
        Dataframe containing only subjects belonging to tight clusters
        and their cluster membership
    """
    clust_totals = clusterdata.apply(pd.value_counts, axis=1)
    clust_pct = clust_totals / len(clusterdata.columns)
    pos_subs = clust_pct.index[clust_pct['pos'] > .96]
    neg_subs = clust_pct.index[clust_pct['neg'] > .96]
    tight_subs = pd.DataFrame(index=pos_subs + neg_subs,columns=pd.Index([0]))
    tight_subs[0].ix[pos_subs] = 1
    tight_subs[0].ix[neg_subs] = 2
    return tight_subs
    

def run_clustering(infile, nperm, weightsum, bound):
    """
    Runs sparse k-means resampling. For each sample, runs a clustering on a
    subset of subjects. These clusters of subjects are used to generate 
    regional cut-offs in order to classify the remaining subjects in the sample.
    
    Inputs
    ------
    infile: str
        path to input datafile. First column should contain subject
        codes and additional columns should correspond to features
    nperm: int
        number of clustering resample runs
    weightsum: float
        percentage of total feature weights to use in calculating cutoffs
        Only features with the highest weights summing to this value will 
        be used.
    bound: str  ['best' or 'sparse']
        Determines which value generated by SKM permutation to use as
        L1 bound in clustering. This tuning parameter determines how weights 
        will be distributed among features. 'best' will give more non-zero 
        weights.   
    
    Returns
    -------
    weight_rslts: pandas Dataframe
        n by p dataframe where n is the number of features and p is
        the number of permutations (nperms). 
        index = feature names, values = feature weights
    clust_rslts: pandas Dataframe
        n by p dataframe where n is the number of subject and p is
        the number of nperms. 
        index = subject codes, values = cluster membership
            
    """
    # make sure we can import sparcl
    import_sparcl()
    # Load data to dataframe
    dataframe = pd.read_csv(infile, sep=None, index_col=0)
    # Create empty frames to hold results of resampling
    weight_rslts, clust_rslts = create_rslts_frame(dataframe) 
    for resamp_run in range(nperm):
        print 'Now starting re-sample run number %s'%(resamp_run)
        # Get random sub-sample (without replacement) of group 
        # to feed into clustering
        # Currently set to 70% of group N
        traindat, testdat = sample_data(dataframe)
        best_L1bound, lowest_L1bound = skm_permute(traindat)
        if bound == 'sparse':
            km_weight, km_clust = skm_cluster(traindat, lowest_L1bound)      
        else:
            km_weight, km_clust = skm_cluster(traindat, best_L1bound)
        samp_clust, sampcutoffs = calc_cutoffs(traindat, 
                                               km_weight, 
                                               weightsum, 
                                               km_clust)
        unsamp_clust = predict_clust(testdat, sampcutoffs)
        # Log weights and cluster membership of resample run
        weight_rslts[resamp_run] = km_weight[0]
        clust_rslts[resamp_run] = pd.concat([samp_clust, unsamp_clust])
    return weight_rslts, clust_rslts
    

def classify_subjects(infile, weightsum, weight_rslts, clust_rslts):
    """
    Classifies subject into PIB+ and PIB- based on regional cut-offs
    derived from sparse k-means clustering. 
    
    Inputs
    ------
    infile: str (filename)
        path to input datafile. First column should contain subject
        codes and additional columns should correspond to features
    weightsum: float
        percentage of total feature weights to use in calculating cutoffs
        Only features with the highest weights summing to this value will 
        be used.
    weight_rslts: pandas Dataframe
        n by p dataframe where n is the number of features and p is
        the number of permutations (nperms). 
        index = feature names, values = feature weights
    clust_rslts: pandas Dataframe
        n by p dataframe where n is the number of subject and p is
        the number of nperms. 
        index = subject codes, values = cluster membership
                    
    Returns
    -------
    all_clust: pandas Dataframe
        Contains cluster membership for all subjects
        index = subject codes passed from input data
        values = predicted cluster membership  
    weight_totals: pandas Dataframe
        Contains average weight for all feature s
        index = feature names
        values = average weight
    grpcutoffs: pandas Dataframe
        Contains PIB value used as cutoff for all features
        index = feature names
        values = value used as cut-off between groups
    """
    # make sure we can import sparcl
    import_sparcl()
    # Load data to dataframe
    dataframe = pd.read_csv(infile, sep=None, index_col=0)
    # Create tight clusters and generate regional cut-offs to 
    # predict remaining subjects
    weight_totals = weight_rslts.mean(axis=1)
    tight_subs = create_tightclust(clust_rslts)
    tight_clust, grpcutoffs = calc_cutoffs(dataframe.ix[tight_subs.index], 
                                            pd.DataFrame(weight_totals), 
                                            weightsum,
                                            tight_subs)
    all_clust = predict_clust(dataframe, grpcutoffs) 
    grpcutoffs.name = 'Cutoff_Value'
    weight_totals.name = 'WeightMean'    
    return all_clust, grpcutoffs, weight_totals
    
def save_results(outdir, all_clust, weight_totals, grpcutoffs):
    """
    Inputs
    ------
    outdir: str
        Location (path) to save output files
    all_clust: pandas Dataframe
        Contains cluster membership for all subjects
        index = subject codes passed from input data
        values = predicted cluster membership  
    weight_totals: pandas Dataframe
        Contains average weight for all features
        index = feature names
        values = average weight
    grpcutoffs: pandas Dataframe
        Contains PIB value used as cutoff for all features
        index = feature names
        values = value used as cut-off between groups    
    
    Returns
    -------
    all_clust_out: str
        path to ClusterResults.csv, contains cluster membership 
        of all subjects
    weight_totals_out: str
        path to FeatureWeights.csv. contains average weight
        for each feature
    grpcutoffs_out: str
        path to CutoffValues.csv, contains cutoff value for
        each feature used to classify subjects
    """
    
    # Save out results of cluster membership and feature weights
    all_clust_out = os.path.join(outdir, 'ClusterResults.csv')
    all_clust.to_csv(all_clust_out, 
                     index=True, 
                     index_label='SUBID', 
                     header=True,sep='\t')
    print 'Cluster membership results saved to %s'%(all_clust_out)
    
    weight_totals_out = os.path.join(outdir, 'FeatureWeights.csv')
    weight_totals.to_csv(weight_totals_out, 
                         index=True, 
                         index_label='Feature', 
                         header=True,sep='\t')
    print 'Mean feature weights saved to %s'%(weight_totals_out)
    
    grpcutoffs_out = os.path.join(outdir, 'CutoffValues.csv')
    grpcutoffs.to_csv(grpcutoffs_out, 
                      index=True, 
                      index_label='Feature', 
                      header=True,sep='\t')
    print 'Regions and cutoff scores used to determine groups saved'\
            'to %s'%(grpcutoffs_out)
    return all_clust_out, weight_totals_out, grpcutoffs_out
    
    
def main(infile, outdir, nperm, weightsum, bound):
    """
    Function to run full script. Takes input generated from command line 
    and runs sparse k-means clustering with resampling to produce tight 
    clusters.
    Regional cutoffs are derived from these tight clusters and subjects
    are classified based on these cutoffs. Saves out results to file.
    """
    weight_rslts, clust_rslts = run_clustering(infile, nperm, weightsum, bound)
    all_clust, grpcutoffs, weight_totals = classify_subjects(infile, 
                                                            weightsum, 
                                                            weight_rslts, 
                                                            clust_rslts)
    (all_clust_out, 
     weight_totals_out,
     grpcutoffs_out) = save_results(outdir, 
                                    all_clust, weight_totals, grpcutoffs)

##########################################################################
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = """
    Sparse K-means Clustering with re-sampling to cluster subjects into 
    PIB+ and PIB- groups.
    -------------------------------------------------------------------""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog= """
    -------------------------------------------------------------------
    NOTE: Format of infile should be a spreadsheet with first column 
    containing subject codes and the remaining columns corresponding to 
    ROIs to be used as features in clustering. The first row may contain 
    headers.""")

    parser.add_argument('infile', type=str, nargs=1,
            help='Input file containing subject codes and PIB indices')
    parser.add_argument('-outdir', dest='outdir', default=None,
            help='Directory to save results. (default = infile directory)')
    parser.add_argument('-nperm', type=int, dest = 'nperm', default = 1000, 
            help = 'Number of re-sample permutations (default 1000)')
    parser.add_argument('-weightsum', type=float, dest = 'weightsum', 
                        default = 1.0, 
            help = """Only determine cutoffs for features with weights making 
            up the top percentile specified. Constrains the number 
            of features used for classification to those that were 
            most important in clustering (default = 1.0)""")
    parser.add_argument('-bound', type=str, choices=['best', 'sparse'],
                        dest='bound', default='best',
            help="""method to determine L1bound, best result, or best  
            sparse result""")
	

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        if args.outdir is None:
            args.outdir, _ = os.path.split(args.infile[0])

            
        ### Begin running SKM clustering and resampling  
        main(args.infile[0], 
             args.outdir, 
             args.nperm, 
             args.weightsum, 
             args.bound)



