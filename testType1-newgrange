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
SnpSize=[2000,1000,100]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(200))

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

DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')

genZScores(parms,[2,3])

N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]

#######################################################################################################

LZCorr3,offDiag=genLZCorr({**parms,'snpChr':[3]})
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(offDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag3.png')
plt.close('all') 

#######################################################################################################

LZCorr2,offDiag=genLZCorr({**parms,'snpChr':[2]})
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(offDiag,bins=60)
fig.savefig('diagnostics/v(z)OffDiag2.png')
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
zDat2=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zNormLZ3=np.matmul(norm.rvs(size=[int(1e6),int(N)]),LZCorr3.T)
zNormLZ2=np.matmul(norm.rvs(size=[int(1e6),int(N)]),LZCorr2.T)
zNormI=norm.rvs(size=[int(1e6),int(N)])

#######################################################################################################

zSet=[zDat2,zDat3,zNormI]
nm=['2','3','I']
for i in [0,1]:
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(zSet[i].flatten()**2)
    x=chi2.ppf(np.arange(1,len(y)+1)/(len(y)+1),1)
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('sq-'+str(nm[i]))
    fig.savefig('diagnostics/z^2 '+str(nm[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i]**2,axis=0).flatten())
    x=chi2.ppf(np.arange(1,zSet[i].shape[1]+1)/(zSet[i].shape[1]+1),zSet[i].shape[0])/zSet[i].shape[0]
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('eta '+str(nm[i]))
    fig.savefig('diagnostics/traitMean z^2 '+str(nm[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i]**2,axis=1).flatten())
    x=chi2.ppf(np.arange(1,zSet[i].shape[0]+1)/(zSet[i].shape[0]+1),zSet[i].shape[1])/zSet[i].shape[1]
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('eta '+str(nm[i]))
    fig.savefig('diagnostics/snpMean z^2 '+str(nm[i])+'.png')
    plt.close('all') 

#######################################################################################################

ell3=stat.score(zDat3)
ell2=stat.score(zDat2)
refLZ2=stat.score(zNormLZ2)
refLZ3=stat.score(zNormLZ3)
refI=stat.score(zNormI)

#######################################################################################################

monteCarlo2_3=stat.monteCarlo(ell2,ell3)
monteCarloI_3=stat.monteCarlo(refI,ell3)
monteCarloLZ2_3=stat.monteCarlo(refLZ2,ell3)
monteCarloLZ3_3=stat.monteCarlo(refLZ3,ell3)

#######################################################################################################

markov3=stat.markov(ell3)
markov2=stat.markov(ell2)

#######################################################################################################

plotPower(monteCarlo2_3,parms,'mc2_3',['mc2_3-'+str(x) for x in ellDSet])
plotPower(monteCarloI_3,parms,'mcI_3',['mcI_3-'+str(x) for x in ellDSet])
plotPower(monteCarloLZ2_3,parms,'mcLZ2_3',['mcLZ2_3-'+str(x) for x in ellDSet])
plotPower(monteCarloLZ3_3,parms,'mcLZ3_3',['mcLZ3_3-'+str(x) for x in ellDSet])
plotPower(markov3,parms,'markov3',['markov3-'+str(x) for x in ellDSet])
plotPower(markov2,parms,'markov2',['markov2-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)

DBFinish(parms)
