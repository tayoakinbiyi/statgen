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
SnpSize=[100,100,100]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(100))

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

genZScores(parms,[2,3])
'''
N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]

#######################################################################################################

DBLog('genLZCorr')
LZCorr,offDiag=genLZCorr({**parms,'snpChr':[3]})
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(offDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag.png')
plt.close('all') 

#######################################################################################################

offDiag=np.array([0]*int(N*(N-1)/2))
stat=ell(np.array(ellDSet),offDiag)

#######################################################################################################

stat.fit(20,1000,2000,8,1e-7) # initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta
stat.save()
#stat.load()
#######################################################################################################

zDat3=np.concatenate([np.loadtxt('score/waldStat-3-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ell3=stat.score(zDat3)

zDat2=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
ell2=stat.score(zDat2)

#######################################################################################################

'''
#zRef=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zNorm=np.matmul(norm.rvs(size=[int(1e6),int(N)]),LZCorr.T)

fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(np.mean(zNorm**2,axis=1),bins=60,density=True,alpha=.5,label='norm')
axs.hist(np.mean(zRef**2,axis=1),bins=60,density=True,alpha=.5,label='ref')
axs.legend()
fig.savefig('diagnostics/snpMeans.png')
plt.close('all') 

fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
y=np.sort(np.mean(zNorm**2,axis=0))
x=np.sort(1+norm.rvs(size=(436,))*np.sqrt(2*1e-6))
axs.scatter(x,y)
mMax=max(np.max(x),np.max(y))
mMin=max(np.min(x),np.min(y))
axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
fig.savefig('diagnostics/traitMeans.png')
plt.close('all') 
'''

#######################################################################################################

zNormLZ=np.matmul(norm.rvs(size=[int(1e6),int(N)]),LZCorr.T)
zNormI=norm.rvs(size=[int(1e6),int(N)])
refLZ=stat.score(zNormLZ)
refI=stat.score(zNormI)
monteCarlo2_3=stat.monteCarlo(ell2,ell3)
monteCarloI_3=stat.monteCarlo(refI,ell3)
monteCarloLZ_3=stat.monteCarlo(refLZ,ell3)
#monteCarlo=stat.monteCarlo(ref,ell)

#######################################################################################################

markov3=stat.markov(ell3)
markov2=stat.markov(ell2)

#######################################################################################################

plotPower(monteCarlo2_3,parms,'mc2_3',['mc2_3-'+str(x) for x in ellDSet])
plotPower(monteCarloI_3,parms,'mcI_3',['mcI_3-'+str(x) for x in ellDSet])
plotPower(monteCarloLZ_3,parms,'mcLZ_3',['mcLZ_3-'+str(x) for x in ellDSet])
plotPower(markov3,parms,'markov3',['markov3-'+str(x) for x in ellDSet])
plotPower(markov3,parms,'markov2',['markov2-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)

DBFinish(parms)
