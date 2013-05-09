import pandas as pd
import os, sys
import argparse


def select_visit(infile, outfile, visit='last'):
    """
    Takes input spreadsheet and loads into pandas DataFrame. Assumes first 
    column contains subject codes. If multiple visits exist, selects indicated 
    datapoint (defaults to last). Returns DataFrame with subject codes as index.
    
    NOTE: Currently, can only determine timepoint if subject codes contain visit 
    indicator (i.e. '_v1')
    """
    data = pd.read_csv(infile, sep=None)
    subj_col = data.columns[0]    
    # Check to see if subject codes contain visit indicator
    if any(data[subj_col].str.contains('B*_v.')):
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
        
    cleandata.to_csv(outfile, sep='\t', index_label='SUBID', header=True)   
    return cleandata


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = """
    Prepares data for use as input to sparse K-means clustering. Assumes first 
    column contains subject codes. If multiple visits exist, selects indicated 
    datapoint (defaults to last). 
    -------------------------------------------------------------------""",
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog= """
    -------------------------------------------------------------------
    NOTE: Currently, can only determine timepoint if subject codes contain visit 
    indicator (i.e. '_v1').""")

    parser.add_argument('infile', type=str, nargs=1,
                        help="""Input file containing subject codes and PIB indices""")
    parser.add_argument('outfile', type=str, nargs=1,
                        help="""Path and name to save output.""")
    parser.add_argument('-visit', type=str, dest='visit', default=None,
                        choices=['first', 'last'],
                        help="""Specify which visit to use if multiple visits exist \ndefault = last""")

                        
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()

        if args.visit is None:
            args.visit = 'last'   
            
        slect_visit(args.infile[0], args.outfile[0], args.visit)                     
