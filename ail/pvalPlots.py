import pandas as pd
import numpy as np
import os
import pdb
from scipy.stats import norm
import matplotlib.pyplot as plt

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
# include cis [0,1], exclude cis [0,1]

allP=np.sort(allRes.values.flatten())
expP=np.arange(1,len(allP)+1)/len(allP)

expP=-np.log(expP[allP<.01])
allP=-np.log(allP[allP<.01])

fig,axs=plt.subplots(1,1)
axs.plot(expP,allP)
axs.plot(expP,expP,color='r')
fig.savefig('t.png')


