import os, sys
import argparse
import pandas as pd
import pandas.rpy.common as com
import rpy2
import random
sparcl = rpy2.robjects.packages.importr("sparcl")


def prepare_data(infile, visit='last'):
    """
    Takes input spreadsheet and loads into pd DataFrame. Assumes first 
    column contains subject codes. If multiple visits exist, selects indicated 
    datapoint (defaults to last). Returns DataFrame with subject codes as index.
    
    NOTE: Currently, can only determine timepoint if subject codes contain visit 
    indicator (i.e. '_v1')
    """
    data = pd.read_csv(infile, sep=None)
    subj_col = data.columns[0]    
    # Check to see if subject codes contain visit indicator
    if any(data.SUBID.str.contains('B*_v.')):
        splitcol = data[subj_col].str.split('_')
        data[subj_col] = splitcol.str[0]
        data['Visit'] = splitcol.str[1]
        # Sort data by subject code and visit number
        sorteddata = data.sort(columns=[subj_col, 'Visit'])       
        # Select either first or last visit depending on args. default is last
        if visit == 'first':
            singlevisit = sorteddata.drop_duplicates(cols=subj_col, take_last=False)
        else:
            singlevisit = sorteddata.drop_duplicates(cols=subj_col, take_last=True)           
        singlevisit = singlevisit.drop('Visit', axis=1)
        cleandata = singlevisit.set_index(subj_col)        
    else:
        singlevisit = sorteddata.drop_duplicates(cols=subj_col, take_last=True)
        cleandata = singlevisit.set_index(subj_col)      
    return cleandata


def create_rslts_frame(dataframe):  
    """ Create empty dataframe to hold eature weights and cluster 
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
    return sampdata


def skm_permute(data):
    """
    rpy2 wrapper for R function: KMeansSparseCluster.permute from the sparcl package.
    The tuning parameter controls the L1 bound on w, the feature weights. A permutation 
    approach is used to select the tuning parameter. 
    
    Infile:
    ---------------
    data: nxp dataframe where n is observations and p is features (i.e. ROIs)
            Should be a pandas DataFrame with subject codes as index and only
            features to be used in clustering as columns.
    
    Returns:
    ---------------
    bestw: tuning parameter that returns the highest gap statistic
    
    lowestw: smallest tuning parameter that gives a gap statistic within 
                one sdgap of the largest gap statistic
    """
    r_data = com.convert_to_r_dataframe(data, L1bound)
    km_perm = sparcl.KMeansSparseCluster_permute(r_data,K=2,nperms=25)
    bestw = km_perm.rx2('bestw')[0]
    wbounds = km_perm.rx2('wbounds')
    gaps = km_perm.rx2('gaps')
    bestgap = max(gaps)
    sdgaps = km_perm.rx2('sdgaps')
    # Calculate smallest wbound that returns gap stat within one sdgap of best wbound
    wbound_rnge = [wbounds[i] for i in range(len(gaps)) if (gaps[i]+sdgaps[i]>=bestgap)]
    lowestw = min(wbound_rnge)
    return bestw, lowestw
    
    
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
            Should be a pandas DataFrame with subject codes as index and only
            features to be used in clustering as columns.
    
    Returns:
    ---------------
    km_weights: pandas DataFrame   
                index = feature labels, values = weights
    km_clusters: pandas DataFrame
                index = subject code, values = cluster membership
    """
    # Convert pandas dataframe to R dataframe
    r_data = com.convert_to_r_dataframe(data)
    # Cluster observations using specified L1 bound
    km_out = sparcl.KMeansSparseCluster(r_data,K=2, wbounds=L1bound)
    # Create dictionary of feature weights, normalized feature weights and
    # cluster membership
    ws = km_out.rx2(1).rx2('ws')
    km_weights = {k: [ws.rx2(k)[0]] for k in ws.names}
    km_weights = pd.DataFrame.from_dict(km_weights)
    km_weightsT = km_weights.T
    km_weightsnorm = km_weightsT/km_weightsT.sum()
    Cs = km_out.rx2(1).rx2('Cs')
    km_clusters = {k: [Cs.rx2(k)[0]] for k in Cs.names}   
    km_clusters = pd.DataFrame.from_dict(km_clusters)
    km_clusters = km_clusters.T
    return km_weightsnorm, km_clusters
        
        
def clust_centers(data, clust_num):
    """
    Determines the cluster centers given an array of data and cluster membership.
    
    Inputs
    ----------
    data: an nxp array with n observations and p features
    clust_num: dict {subject id: cluster number}
    
    Returns:
    ----------
    clust1_means: array listing the feature label and mean across cluster 1 members
    clust2_means: array listing the feature label and mean across cluster 2 members
    """
    clust1_idx = [clust_num[key]==1 for key in clust_num.keys()]
    clust1_means = data[clust1_idx].mean(axis=0)
    clust2_idx = [clust_num[key]==2 for key in clust_num.keys()]
    clust2_means = data[clust2_idx].mean(axis=0)
    return clust1_means, clust2_means


    

def main(infile, outdir, visit):
    # Select one visit, clean subj codes
    dataframe = prepare_data(infile, visit) 
    # Create empty frames to hold results of 
    weight_rslts, clust_rslts = create_rslts_frame(dataframe) 
    for resamp_run in range(10):
        # Get random sub-sample (without replacement) of group to feed into clustering
        # Currently set to 70% of group N
        sampdat = subsample_data(dataframe)
        bestw, lowestw = skm_permute(sampdat)
        km_wghtnorm, km_clust = skm_cluster(sampdat, lowestw)      
        clust1_means, clust2_means = clust_centers(sampdat, km_clust)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = """Sparse K-means Clustering with re-sampling to cluster subjects into PIB+ and PIB- groups.""",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog= """Format of infile should be a spreadsheet with first column containing subject ID's and the remaining columns corresponding to ROIs to be used as features in clustering. The first row may contain headers.""")

    parser.add_argument('infile', type=str, nargs=1,
                        help="""Input file containing subject codes and PIB indices""")
    parser.add_argument('-outdir', dest='outdir', default=None,
                        help="""Directory to save results. \ndefault = infile directory""")
    parser.add_argument('-visit', type=str, dest='visit', default=None,
                        choices=['first', 'last'],
                        help="""Specify which visit to use if multiple visits exist \ndefault = last""")
    
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        if args.outdir is None:
            args.outdir, _ = os.path.split(args.infile[0])
        if args.visit is None:
            args.visit = 'last'
            
        ### Begin running SKM clustering and resampling  
        main(args.infile, args.outdir, args.visit)


    # Load input spreadsheet into dataframe and select visit if multiple exist
    datclean = prepare_data(infile)
    datframe = pd.read_csv(infile, sep=None)
    # Strip visit id (i.e. '_v1') from subject codes if they exist.
    # Take most recent visit if multiple are included.
    datclean = select_visit(datframe)
    
    
    subjects = datframe.index
    features = datframe.columns
    
    resamp_weights = pd.DataFrame(data=None, columns = features)
    resamp_clusters = pd.DataFrame(data=None, columns = subjects)
    
    for resamp_run in range(10):
        # Get random sub-sample (without replacement) of group to feed into clustering
        # Currently set to 70% of group N
        sampdat = subsample_data(datframe)

        # Convert sampled data to R dataframe and run skm clustering
        r_sampdat = com.convert_to_r_dataframe(sampdat)
        km_wghtnorm, km_clust = skm_cluster(r_sampdat)
        
        # Find cluster centers for each group
        clust1_means, clust2_means = clust_centers(sampdat, km_clust)

        # Predict cluster membership for remaining subjects
        """
        for subj in subjects:
            if subj not in sampdat.index:
                compare subj data to clust1_means and clust2_means
                or
                if any region surpasses cut-off, then PIB+
        """          
        
                
        # Append results of resample to results arrays
        resamp_weights = resamp_weights.append(km_wghtnorm, ignore_index=True)    
        resamp_clusters = resamp_clusters.append(km_clust, ignore_index=True)
    

   
"""
TO DO:
--------
1. Clean up input file. Check for subject id's with '_v1' etc. If they exist, 
take most recent scan and strip visit indicator.
2. Check conversion from dataframe to R. doesn't seem to match up.
3. Predict cluster membership of remaining 30% of subjects and 
save to results data arrays
4. Construct tight clusters using subjects assigned to a cluster 
in >=96% of samples
5. Mid-point of these clusters becomes cut-off, a subject with 
PIB index above this cut-off in any cluster is considered PIB+
""" 
    
    
