import matplotlib
matplotlib.use('agg')
import numpy as np
import os
import pdb
import sys

sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *
from multiprocessing import cpu_count
from plotPython.plotPower import *

import subprocess
from scipy.stats import norm

from ELL.ell import *

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[10000,10000,10000]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

ctrl={
    'etaSq':0,
    'numSubjects':208*3,
    'YTraitIndep':'indep',#['indep','dep','real']
    'modelTraitIndep':'indep',#['indep','dep']
    'fastlmm':False
}
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'colors':colors,
    'refReps':1e6,    
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'grm':'gemmaStd',#['gemmaNoStd','gemmaStd','fast']
    'traitSubset':traitSubset,
    'maxSnpGen':5000,
    'transOnly':False
}

parms=setupFolders(ctrl,ops)
'''
DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')

genZScores(parms)
'''
N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]

DBLog('genLZCorr')
genLZCorr({**parms,'snpChr':[2]})
LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
zOffDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]

#######################################################################################################
'''
offDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]
stat=ell(N,np.array(ellDSet)*N,parms['numCores'],True)

#######################################################################################################

stat.fit(20,1000,2000,8,1e-7,offDiag) # initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta
stat.save()
#stat.load()
#######################################################################################################

zDat=np.concatenate([np.loadtxt('score/waldStat-3-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ell=stat.score(zDat)

#######################################################################################################

fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(zOffDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag.png')
plt.close('all') 
'''
#zRef=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zNorm=np.matmul(norm.rvs(size=[int(1e6),int(N)]),LZCorr.T)
'''
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(np.mean(zNorm**2,axis=1),bins=60,density=True,alpha=.5,label='norm')
axs.hist(np.mean(zRef**2,axis=1),bins=60,density=True,alpha=.5,label='ref')
axs.legend()
fig.savefig('diagnostics/snpMeans.png')
plt.close('all') 
'''
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
y=np.sort(np.mean(zNorm**2,axis=0))
x=np.sort(1+norm.rvs(size=(436,))*np.sqrt(2*1e-6))
axs.scatter(x,y)
mMax=max(np.max(x),np.max(y))
axs.plot([0,mMax], [0,mMax], ls="--", c=".3")   
fig.savefig('diagnostics/traitMeans.png')
plt.close('all') 
'''
ref=stat.score(zNorm)

#######################################################################################################

monteCarlo=stat.monteCarlo(ref,ell)

#######################################################################################################

markov=stat.markov(ell,offDiag)

#######################################################################################################

plotPower(monteCarlo,parms,'mc',['mc-'+str(x) for x in ellDSet])
plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])
pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)
'''
DBFinish(parms)
