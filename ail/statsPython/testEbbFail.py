import warnings
import matplotlib
matplotlib.use('agg')

from genPython.makePSD import *
from statsPython.makeProb import *

import numpy as np
import pdb
import os  

home='/deeplearning/akinbiyi/'
#home='/project/abney/'
#home='/home/ubuntu/ail/'
#home='/home/tayo/ail/'
#home='/home/akinbiyi/ail/'

N=500
rho=.1
sigDir='testEBB-'+str(N)+'/'

parms={
    'dataDir':home+'data/',
    'scratchDir':home+'scratch/'+sigDir,
    'plotsDir':home+'ail/plots/'+sigDir,
    'gemma':home+'gemma',
    'plot':True,
    'H0':20,
    'fontsize':17,
    'new':True,
    'Rpath':home+'ggnull/R',
    'cpu':cpu_count(),
    'binPower':500,
    'eps':1e-10,
    'delta':.5,
    'H0':30,
    'N':N
}
#np.corrcoef(norm.rvs(size=[200,N]),rowvar=False)

scratchDir=parms['scratchDir']    
plotsDir=parms['plotsDir']

if not os.path.exists(scratchDir):
    os.mkdir(scratchDir)
    
if not os.path.exists(plotsDir):
    os.mkdir(plotsDir)

sig=np.ones(N)*rho+(1-rho)*np.eye(N)
sigOld=np.loadtxt('/deeplearning/akinbiyi/sig.csv',delimiter=',')
L=makePSD(sig)
LOld=np.loadtxt('/deeplearning/akinbiyi/L.csv',delimiter=',')
sig=np.matmul(L,L.T)
    
fig,axs=plt.subplots(1,1)
fig.set_figwidth(7,forward=True)
fig.set_figheight(7,forward=True)
off_diag=sig[np.triu_indices(N,1)].flatten()  
pairwise_cors=np.loadtxt('/deeplearning/akinbiyi/pairwise_cors.csv',delimiter=',').flatten()

axs.hist(off_diag,bins=np.linspace(-1,1,100))
fig.savefig(plotsDir+'off_diag_hist.png',bbox_inches='tight')
np.savetxt(scratchDir+'pairwise_cors.csv',pairwise_cors,delimiter=',')

plt.close()    
    
ggnullDat,ghcDat=makeProb(L,parms)

fail.to_csv(scratchDir+'fail',index=False)

print(fail.shape[0],flush=True)