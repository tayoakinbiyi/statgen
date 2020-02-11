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
n=3
m=10
ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
snpSize=[208*n*m]*3
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(snpSize)+1)]
traitSubset=list(range(500))

ctrl={
    'etaSq':0,
    'numSubjects':208*n,
    'YType':'simIndep',#['simDep','real','simIndep']
    'snpType':'sim',#['real','sim','random']
    'modelTraitIndep':'indep',#['indep','dep']
    'lmm':'gemma-lm', #['gemma-lmm','gemma-lm','fastlmm']
    'grm':'gemmaStd',#['gemmaNoStd','gemmaStd','fast']
    'normalize':'quant',#['quant','none','std']
    'snpSize':snpSize
}
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'colors':colors,
    'refReps':1e6,    
    'simLearnType':'Full',
    'response':'hipRaw',
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'traitSubset':traitSubset,
    'maxSnpGen':5000,
    'transOnly':False
}

#######################################################################################################

parms=setupFolders(ctrl,ops)

DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

DBLog('makeSimPedFiles')
makeSimPedFiles(parms)

DBLog('genZScores')

genZScores({**parms,'snpChr':[1,2,3]})

N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]
numSnps=int(pd.read_csv('ped/snpData',sep='\t',index_col=None,header=0).shape[0]/3)

#######################################################################################################

zDat3=np.concatenate([np.loadtxt('score/waldStat-3-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zDat2=np.concatenate([np.loadtxt('score/waldStat-2-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zDat1=np.concatenate([np.loadtxt('score/waldStat-1-'+str(x),delimiter='\t') for x in traitChr],axis=1)
zDatI=norm.rvs(size=[len(zDat3),int(N)])
zRef=norm.rvs(size=[int(1e6),int(N)])
Y=np.loadtxt('ped/Y.txt',delimiter='\t')

#######################################################################################################

zSet=[zDat1,zDat2,zDat3,zDatI]
nm=['1','2','3','I']
for i in range(len(nm)):
    print(i,nm[i],zSet[i].shape)
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i],axis=0))
    x=norm.ppf(np.arange(1,N+1)/(N+1))*np.std(y)
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('traitMean-'+str(nm[i]))
    fig.savefig('diagnostics/traitMean '+str(nm[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i],axis=1))
    x=norm.ppf(np.arange(1,numSnps+1)/(numSnps+1))*np.std(y)
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('snpMean-'+str(nm[i]))
    fig.savefig('diagnostics/snpMean '+str(nm[i])+'.png')
    plt.close('all') 

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
    x=chi2.ppf(np.arange(1,N+1)/(N+1),numSnps)/numSnps
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
    x=chi2.ppf(np.arange(1,numSnps+1)/(numSnps+1),N)/N
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('eta '+str(nm[i]))
    fig.savefig('diagnostics/snpMean z^2 '+str(nm[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    axs.hist(np.corrcoef(zSet[i],rowvar=False)[np.triu_indices(N,1)],bins=60)
    fig.savefig('diagnostics/offDiag-'+str(nm[i])+'.png')
    plt.close('all') 
    
fig,axs=plt.subplots(1,1)
fig.set_figwidth(10,forward=True)
fig.set_figheight(10,forward=True)
axs.hist(np.corrcoef(Y,rowvar=False)[np.triu_indices(N,1)],bins=60)
fig.savefig('diagnostics/offDiag-Y.png')
plt.close('all') 

#######################################################################################################

offDiag=np.array([0]*int(N*(N-1)/2))
stat=ell(np.array(ellDSet),offDiag)

#######################################################################################################

stat.load()
if stat.N!=N or np.sum(stat.offDiag)!=np.sum(offDiag):
    stat.fit(10*N,1000*N,3000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
    
#######################################################################################################

ell3=stat.score(zDat3)
ell2=stat.score(zDat2)
ell1=stat.score(zDat1)
ellI=stat.score(zDatI)
ref=stat.score(zRef)

#######################################################################################################

monteCarlo3=stat.monteCarlo(ref,ell3)
monteCarlo2=stat.monteCarlo(ref,ell2)
monteCarlo1=stat.monteCarlo(ref,ell1)
monteCarloI=stat.monteCarlo(ref,ellI)

#######################################################################################################

markov3=stat.markov(ell3)
markov2=stat.markov(ell2)
markov1=stat.markov(ell1)

#######################################################################################################

plotPower(monteCarloI,parms,'mcI',['mcI-'+str(x) for x in ellDSet])
plotPower(monteCarlo3,parms,'mc3',['mc3-'+str(x) for x in ellDSet])
plotPower(monteCarlo2,parms,'mc2',['mc2-'+str(x) for x in ellDSet])
plotPower(monteCarlo1,parms,'mc1',['mc1-'+str(x) for x in ellDSet])
plotPower(markov3,parms,'markov3',['markov3-'+str(x) for x in ellDSet])
plotPower(markov2,parms,'markov2',['markov2-'+str(x) for x in ellDSet])
plotPower(markov1,parms,'markov1',['markov1-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)

DBFinish(parms)
