import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys

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

ctrl={
    'numSubjects':208*3,
    'YTraitIndep':'indep',#['indep','dep','real']
    'modelTraitIndep':True,
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
    'fastlmm':True,
    'grm':2,#[1,2,'fast']
    'traitSubset':traitSubset,
    'maxSnpGen':5000,
    'transOnly':False
}

parms=setupFolders(ctrl,ops)
zSet=[]

#######################################################################################################

DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

#######################################################################################################

for eta in etaSet:
    parms['etaSq']=eta
    
    DBLog('makeSimPedFiles')
    makeSimPedFiles(parms)

    DBLog('genZScores')
    genZScores(parms)
    
    subprocess.call(['cp','-f','score/waldStat-3-18','waldStat-3-18-'+str(eta)])

for eta in etaSet:
    zSet+=[np.loadtxt('waldStat-3-18-'+str(eta),delimiter='\t')]

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

for i in range(len(offDiagSet)):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    axs.hist(offDiagSet[i],bins=30)
    axs.set_title(str(etaSet[i]))
    fig.savefig('diagnostics/hist-'+str(etaSet[i])+'.png')
    plt.close('all') 

#######################################################################################################

for i in range(len(zSet)):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(zSet[i].flatten()/np.std(zSet[i]))
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('eta '+str(etaSet[i]))
    fig.savefig('diagnostics/eta-'+str(etaSet[i])+'.png')
    plt.close('all') 
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i],axis=0))
    y/=np.std(y)
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('trait means eta '+str(etaSet[i]))
    fig.savefig('diagnostics/traitMeans '+str(etaSet[i])+'.png')
    plt.close('all') 

    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    y=np.sort(np.mean(zSet[i],axis=1))
    y/=np.std(y)
    x=norm.ppf(np.arange(1,len(y)+1)/(len(y)+1))
    axs.scatter(x,y)
    mMax=max(np.max(y),np.max(x))
    mMin=min(np.min(y),np.min(x))
    axs.plot([mMin,mMax], [mMin,mMax], ls="--", c=".3")   
    axs.set_title('snp means eta '+str(etaSet[i]))
    fig.savefig('diagnostics/snpMeans '+str(etaSet[i])+'.png')
    plt.close('all') 

#######################################################################################################

for i in range(len(zSet)):
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    uu=zSet[i].flatten()**2
    axs.hist(uu,bins=30)
    axs.set_title('eta '+str(etaSet[i]))
    fig.savefig('diagnostics/z^2 '+str(etaSet[i])+'.png')
    plt.close('all') 
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    uu=np.sum(zSet[i]**2,axis=0).flatten()/np.std(zset[i],axis=0)
    axs.hist(uu,bins=30)
    axs.set_title('eta '+str(etaSet[i]))
    fig.savefig('diagnostics/z^2 '+str(etaSet[i])+'.png')
    plt.close('all') 

DBFinish(parms)
