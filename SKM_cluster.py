import os, sys
import argparse
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
    if not sparclIinstalled():
        install_sparcl()
    sparcl_dir = get_sparcl_dir()
    sparcl = rpy2.robjects.packages.importr("sparcl", lib_loc=sparcl_dir)
    return sparcl


def create_rslts_frame(dataframe):  
    """ Create empty pandas dataframe to hold feature weights and cluster 
    membership results of each resampling run""" 
    weight_rslts = pd.DataFrame(data=None, index = dataframe.columns)
    clust_rslts = pd.DataFrame(data=None, index = dataframe.index)
    return weight_rslts, clust_rslts
    
    
def sample_data(data):
    """
    Takes an array of data as input. Randomly samples 70% of the observations
    and returns as an array
    """
    samp_n = int(.70 * len(data))
    rand_samp = sorted(random.sample(xrange(len(data)), samp_n))
    sampdata = data.take(rand_samp)
    unsamp_idx = [x for x in xrange(len(data)) if x not in rand_samp]
    unsampdata = data.take(unsamp_idx)
    return sampdata, unsampdata


def skm_permute(data):
    """
    rpy2 wrapper for R function: KMeansSparseCluster.permute from the sparcl package.
    The tuning parameter controls the L1 bound on w, the feature weights. A permutation 
    approach is used to select the tuning parameter. 
    
    Infile:
    ---------------
    data: pandas Dataframe
            nxp dataframe where n is observations and p is features (i.e. ROIs)
            Should be a pandas DataFrame with subject codes as index and features 
            as columns.
    
    Returns:
    ---------------
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
    # Calculate smallest wbound that returns gap stat within one sdgap of best wbound
    wbound_rnge = [wbounds[i] for i in range(len(gaps)) if (gaps[i]+sdgaps[i]>=bestgap)]
    lowest_L1bound = min(wbound_rnge)
    return best_L1bound, lowest_L1bound
    
    
def skm_cluster(data, L1bound):
    """
    rpy2 wrapper for R function: KMeansSparseCluster from the sparcl package.
    This function performs sparse k-means clustering. You must specify L1 bound on w, 
    the feature weights. 
    
    Note: A smaller L1 bound will results in sparser weighting. If a large number of 
    features are included, it may be useful to use the smaller tuning parameter 
    returned by KMeansSparseCluster.permute wrapper function.
    
    Infile:
    ---------------
    data: nxp dataframe where n is observations and p is features (i.e. ROIs)
            Should be a pandas DataFrame with subject codes as index and features 
            as columns.
    
    Returns:
    ---------------
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
        
        
def calc_cutoffs(data, weights, weightsum, clusters):
    """
    Determines the cluster centers for regions. This will only detmine cutoffs 
    for features with weights making up the top percentile specified. Clusters 
    are then labelled as positive or negative based on which has the highest 
    average PIB index across these features. This is done because clustering 
    does not consistently assign the same label to groups across runs. 
    
    Inputs
    ----------
    data:   pandas Dataframe
            nxp dataframe where n is observations and p is features (i.e. ROIs)
            Should be a pandas DataFrame with subject codes as index and features 
            as columns.
            
    weights:    pandas Dataframe   
                index = feature labels, values = weights
    weightsum:  float
                cluster cut-offs will be determined for only for features with weihts
                in the top percentile specified
    clusters:   pandas Dataframe
                index = subject code, values = cluster membership
    
    Returns:
    ----------
    clust_renamed:  pandas Dataframe
                    renamed version of clusters input such that integer labels
                    are replaced with appropriate string label ('pos' or 'neg')
    
    cutoffs:    pandas Dataframe
                index = features labels accounting for to 50% of weights
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
    if clust1mean.mean() > clust2mean.mean():
        pos_means = clust1mean
        neg_means = clust2mean
        clust_renamed = clusters[0].astype(str).replace(['1','2'], ['pos','neg'])
    elif clust1mean.mean() < clust2mean.mean():
        pos_means = clust2mean
        neg_means = clust1mean
        clust_renamed = clusters[0].astype(str).replace(['1','2'], ['neg','pos'])
    cutoffs = (pos_means + neg_means) / 2
    clust_renamed.name = 'PIB_Status'
    return clust_renamed, cutoffs


def predict_clust(data, cutoffs):
    """
    Predict cluster membership for set of subjects using feature cutoff scores.
    Classifies as postive if any feature surpasses cut-off value.
    
    Inputs:
    -------------
    data: nxp dataframe where n is observations and p is features (i.e. ROIs)
            Should be a pandas DataFrame with subject codes as index and features 
            as columns.
            
    cutoffs:   pandas DataFrame   
                index = features of interest labels, values = weights 
    
    Returns:
    predicted_clust:    pandas DataFrame
                        index = subject codes passed from input data
                        values = predicted cluster membership     
    """
    # Check all features against cut-offs
    cutdata = data[cutoffs.index] > cutoffs 
    # Classify as pos if any feature is above cutoff
    cutdata_agg = cutdata.any(axis=1)
    predicted_clust = cutdata_agg.astype(str).replace(['T', 'F'], ['pos', 'neg'])
    predicted_clust.name = 'PIB_Status'
    return predicted_clust
    
    
def create_tighclust(clusterdata):
    """
    Creates tight clusters consisting of subjects classified as a member of given 
    cluster in at least 96% of resample runs.
    Inputs:
    ------------
    clusterdata: Dataframe containing cluster membership over all resamples
                    index = subject code 
                    columns = resample run 
                    values = cluster membership
    Returns:
    ------------
    tight_subs: Dataframe containing only subjects belonging to tight clusters
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
    
    
def main(infile, outdir, nperm, weightsum, bound):
    """
    Function to run full script. Takes input from command line and runs 
    sparse k-means clustering with resampling to produce tight clusters.
    Regional cutoffs are derived from these tight clusters and subjects
    are classified based on these cutoffs.
    """
    # make sure we can import sparcl
    import_sparcl()
    # Load data to dataframe
    dataframe = pd.read_csv(infile, sep=None, index_col=0)
    # Create empty frames to hold results of 
    weight_rslts, clust_rslts = create_rslts_frame(dataframe) 
    for resamp_run in range(nperm):
        print 'Now starting re-sample run number %s'%(resamp_run)
        # Get random sub-sample (without replacement) of group to feed into clustering
        # Currently set to 70% of group N
        sampdat, unsampdat = sample_data(dataframe)
        best_L1bound, lowest_L1bound = skm_permute(sampdat)
	if bound == 'sparse':
            km_weight, km_clust = skm_cluster(sampdat, lowest_L1bound)      
	else:
	    km_weight, km_clust = skm_cluster(sampdat, best_L1bound)
	samp_clust, sampcutoffs = calc_cutoffs(sampdat, km_weight, weightsum, km_clust)
        unsamp_clust = predict_clust(unsampdat, sampcutoffs)
        # Log weights and cluster membership of resample run
        weight_rslts[resamp_run] = km_weight[0]
        clust_rslts[resamp_run] = pd.concat([samp_clust, unsamp_clust])

    # Create tight clusters and generate regional cut-offs to predict remaining subjects
    weight_totals = weight_rslts.mean(axis=1)
    tight_subs = create_tighclust(clust_rslts)
    tight_clust, grpcutoffs = calc_cutoffs(dataframe.ix[tight_subs.index], 
                                            pd.DataFrame(weight_totals), 
                                            weightsum,
                                            tight_subs)
    all_clust = predict_clust(dataframe, grpcutoffs) 
    grpcutoffs.name = 'Cutoff_Value'
    weight_totals.name = 'WeightMean'

    # Save out results of cluster membership and feature weights
    all_clust_out = os.path.join(outdir, 'ClusterResults.csv')
    all_clust.to_csv(all_clust_out, index=True, 
                        index_label='SUBID', header=True,sep='\t')
    print 'Cluster membership results saved to %s'%(all_clust_out)
    weight_totals_out = os.path.join(outdir, 'FeatureWeights.csv')
    weight_totals.to_csv(weight_totals_out, index=True, 
                            index_label='Feature', header=True,sep='\t')
    print 'Mean feature weights saved to %s'%(weight_totals_out)
    grpcutoffs_out = os.path.join(outdir, 'CutoffValues.csv')
    grpcutoffs.to_csv(grpcutoffs_out, index=True, 
                            index_label='Feature', header=True,sep='\t')
    print 'Regions and cutoff scores used to determine groups saved to %s'%(grpcutoffs_out)

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
        main(args.infile[0], args.outdir, args.nperm, args.weightsum, args.bound)



