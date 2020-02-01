import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys

local=os.getcwd()+'/'
sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from simPython.genSimZScores import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *

from statsPython.setupELL import *
from statsPython.genELL import*
from statsPython.makeELLMCPVals_saveZ import*
from statsPython.makeELLMarkovPVals import*
from statsPython.makeGBJPValsSimple import *

from plotPython.plotPower import *
from genPython.makePSD import *

from datetime import datetime
import shutil
import subprocess
import psutil

from ELL.ell import *

warnings.simplefilter("error")

etaSet=[0,.5,.75]
ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[100,100,100]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

parms={
    'file':sys.argv[0],
    'ellBetaPpfEps':1e-12,
    'ellKRanLowEps':.1,
    'ellKRanHighEps':1.9,
    'ellKRanNMult':.4,
    'ellDSet':ellDSet,
    'minELLDecForInverse':4,
    'binsPerIndex':500,
    'local':local,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'refReps':1e6,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'fastlmm':True,
    'grm':2,#[1,2,'fast']
    'traitSubset':traitSubset,
    'numSubjects':208*3,
    'indepTraits':True
}

setupFolders(parms)
zSet=[]

#######################################################################################################

for eta in etaSet:
    parms['etaGRM']=eta
    parms['etaError']=1-eta
    
    #DBCreateFolder('diagnostics',parms)
    #DBCreateFolder('ped',parms)
    #DBCreateFolder('score',parms)
    #DBCreateFolder('grm',parms)

    DBLog('makeSimPedFiles')
    makeSimPedFiles(parms)

    DBLog('genZScores')
    #DBCreateFolder('holds',parms)
    genZScores(parms)

    zSet+=[np.loadtxt('score/waldStat-3-18',delimiter='\t')]

#######################################################################################################

offDiagSet=[np.corrcoef(z,rowvar=False)[np.triu_indices(z.shape[1],1)] for z in zSet]
for i in range(len(offDiagSet)-1):
    for j in range(i+1,len(offDiagSet)):
        fig,axs=plt.subplots(1,1)
        fig.set_figwidth(10,forward=True)
        fig.set_figheight(10,forward=True)
        axs.scatter(offDiagSet[i],offDiagSet[j])
        axs.set_title('x='+str(etaSet[i])+' y='+str(etaSet[j]))
        fig.savefig('diagnostics/x='+str(etaSet[i])+',y='+str(etaSet[j])+'.png')
        plt.close('all') 
        
#######################################################################################################

for i in range(len(zSet)):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(zSet[i].flatten())
    y/=np.std(y)
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    axs.set_title('eta '+str(etaSet[i]))
    fig.savefig('eta '+str(etaSet[i])+'.png')
    plt.close('all') 
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.mean(zSet[i],axis=0)
    y/=np.std(y)
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    axs.set_title('trait means eta '+str(etaSet[i]))
    fig.savefig('eta '+str(etaSet[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.mean(zSet[i],axis=1)
    y/=np.std(y)
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    axs.set_title('snp means eta '+str(etaSet[i]))
    fig.savefig('eta '+str(etaSet[i])+'.png')
    plt.close('all') 
    
DBCreateFolder('pvals',parms)

#######################################################################################################

N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]
L=np.eye(N)
offDiag=np.matmul(L,L.T)[np.triu_indices(N,1)]
stat=ell(offDiag,N,np.array([0.1])*N,reportMem=True)

#######################################################################################################

stat.fit(10,700,700,15,15) # initialNumLamPoints,finalNumLamPoints, numEllPoints,lamZeta,ellZeta

#######################################################################################################

zRef=-np.sort(-np.abs(np.matmul(norm.rvs(size=[int(parms['refReps']),N]),L.T)))   
ref=stat.score(zRef)

#######################################################################################################

for etaInd in range(len(etaSet)): 
    ell=stat.score(zSet[etaInd])
    monteCarlo=stat.monteCarlo(ref,ell)
    markov=stat.markov(ell)
    pvals=pd.DataFrame(-np.log10(np.sort(np.concatenate([monteCarlo,markov],axis=1),axis=0)),
        columns=[nm+'-'+str(x) for nm in ['mc','markov'] for x in ellDSet])
    plotPower(pvals)
    pvals.quantile([.05,.01],axis=0).to_csv('diagnostics/exact-'+str(etaSet[etaInd])+'.csv',index=False)

DBFinish(parms)
