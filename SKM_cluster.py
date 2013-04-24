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

def create_tots_frame(dataframe):  
    """ Create empty dataframe to hold results of feature weights and cluster 
    membership from resampling""" 
    weight_tots = pd.DataFrame(data=None, columns = dataframe.columns)
    clust_tots = pd.DataFrame(data=None, columns = dataframe.index)
    return weight_tots, clust_tots
    
def sample_data(data):
    """
    Takes an array of data as input. Randomly samples 70% of the observations
    and returns as an array
    """
    samp_n = int(.70 * len(data))
    rand_samp = sorted(random.sample(xrange(len(data)), samp_n))
    sampdata = data.take(rand_samp)
#    unsamp_idx = [x for x in xrange(len(data)) if x not in rand_samp]
#    unsamp = data[unsampidx]
    return sampdata

def skm_cluster(data):
    """
    Runs sparse k-means clustering on an nxp data matrix where n is the 
    number of observations, and p is the number of features. First, runs
    a permutation to determine the best wbound - the L1 bound on the feature 
    weights (lower value results in sparse.  
    
    Returns:
    km_weights: dictionary 
                keys = feature labels, values = weights
    km_clusters: dictionary
                keys = subject id, values = cluster membership
    """
    
    # Run permutation on subset in order to generate best wbound
    km_perm = sparcl.KMeansSparseCluster_permute(data,K=2,nperms=25)
    bestw = km_perm.rx2('bestw')
    
    # Cluster observations using best wbound
    km_out = sparcl.KMeansSparseCluster(data,K=2, wbounds=bestw)
    
    # Create dictionary of feature weights, normalized feature weights and
    # cluster membership
    ws = km_out.rx2(1).rx2('ws')
    km_weights = {k: ws.rx2(k)[0] for k in ws.names}
    weightsum = np.sum(km_weights.values())
    km_weightsnorm = {k: km_weights[k] / weightsum for k in km_weights.viewkeys()}
    Cs = km_out.rx2(1).rx2('Cs')
    km_clusters = {k: Cs.rx2(k)[0] for k in Cs.names}   
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
    dataframe = prepare_data(infile, visit) 
    weight_tots, clust_tots = create_tots_frame(dataframe)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Sparse K-means Clustering with re-sampling to cluster subjects into PIB+ and PIB- groups.',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog= """ Format of infile should be a spreadsheet with first column containing subject ID's and the remaining columns corresponding to ROIs to be used as features in clustering. The first row may contain headers.
            """)

    parser.add_argument('infile', type=str, nargs=1,
                        help="""Input file containing subject codes and PIB indices""")
    parser.add_argument('-outdir', dest='outdir', default=None,
                        help="""Directory to save results.
    default = infile directory""")
    parser.add_argument('-visit', type=str, dest='visit', default=None,
                        choices=['first', 'last'],
                        help="""Specify which visit to use if multiple visits exist
    default = last""")
    
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
    
    
