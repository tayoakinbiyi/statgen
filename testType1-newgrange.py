import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys

local=os.getcwd()+'/'
sys.path=[local+'source']+sys.path

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
from distutils.dir_util import copy_tree

warnings.simplefilter("error")

ellDSet=[.25]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[10000,10000,800]
traitChr=[18]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
traitSubset=list(range(1000))

parms={
    'file':sys.argv[0],
    'etaGRM':0,
    'etaError':1,
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
    'RefReps':100000,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'fastlmm':True,
    'grm':2,#[1,2,'fast']
    'traitSubset':traitSubset
}

setupFolders(parms)
'''
DBCreateFolder('diagnostics',parms)
DBCreateFolder('ped',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)
DBCreateFolder('refs',parms)

for independence in [True,False]:
    parms['eyeTrait']=independence
    parms['numSubjects']=208*3
    
    DBLog('makeSimPedFiles')
    makeSimPedFiles(parms)

    DBLog('genZScores')
    DBCreateFolder('holds',parms)
    genZScores(parms)
    
    N=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0).shape[0]
        
    DBLog('genLZCorr')
    genLZCorr({**parms,'snpChr':[2]})
    
    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')
    zOffDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)]
    
    LTraitCorr=np.loadtxt('LZCorr/LTraitCorr',delimiter='\t')
    yOffDiag=np.matmul(LTraitCorr,LTraitCorr.T)[np.triu_indices(N,1)]
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    axs.hist(zOffDiag,bins=30)
    fig.savefig('diagnostics/zOffDiag-'+str(independence)+'.png')
    plt.close('all') 
    
    fig,axs=plt.subplots(1,1)
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    axs.scatter(zOffDiag,yOffDiag)
    fig.savefig('diagnostics/zOffDiagVsyOffDiag-'+str(independence)+'.png')
    plt.close('all') 
    
    for vzType in [2,'I']:
        name=str(vzType)+'-'+str(independence)
        print(name,flush=True)
        
        if vzType=='I':
            np.savetxt('LZCorr/LZCorr',np.eye(N),delimiter='\t')
        
        DBCreateFolder('ELL',parms)
        DBCreateFolder('stats',parms)
        DBCreateFolder('pvals',parms)
        
        DBLog('setupELL')
        setupELL(parms)

        DBLog('genELL')
        np.savetxt('refs/ell-'+name,genELL(parms),delimiter='\t')

        DBCreateFolder('pvals',parms)
        
        
        DBLog('makeELLMCPVals')
        np.savetxt('refs/ref-'+name,makeELLMCPVals(parms),delimiter='\t')
        makeELLMarkovPVals(parms)

        DBLog('plotPower')
        plotPower(parms)
        subprocess.call(['mv','diagnostics/power.png','diagnostics/power-'+name+'.png'])
        subprocess.call(['mv','diagnostics/exact.csv','diagnostics/exact-'+name+'.csv'])
'''
refFiles=os.listdir('refs')
fig,axs=plt.subplots(len(refFiles),1)
fig.set_figwidth(20,forward=True)
fig.set_figheight(10*len(refFiles),forward=True)

for fileInd in range(len(refFiles)):
    p=-np.log10(np.clip(np.loadtxt('refs/'+refFiles[fileInd],delimiter='\t'),1e-30,1))    
    axs[fileInd].boxplot(p,labels=np.arange(p.shape[1]),sym='.')
    axs[fileInd].set_title(refFiles[fileInd])

fig.savefig('diagnostics/refs.png')
plt.close('all') 

DBFinish(parms)
