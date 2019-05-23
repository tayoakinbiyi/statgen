import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
from collections import Counter

from python.oneChrFunc import *

files={
    'dataDir':'/phddata/akinbiyi/ail/data/',
    'scratchDir':'/phddata/akinbiyi/ail/scratch/',
    'gemma':'/phddata/akinbiyi/ail/gemma'
}
numPCs=10


#load raw files
dataDir=files['dataDir']
scratchDir=files['scratchDir']

allRes=pd.read_csv(dataDir+'allRes.csv',index_col=[0,1],header=[0,1,2])

# qq plots
# one row per snp, from chr12, maybe 20 SNPS
# four columns
# include cis [0,1], exclude cis [0,1], include cis [0,.01], exclude cis [0,.01]
# 
hist
        
