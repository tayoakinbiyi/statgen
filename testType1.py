import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
from multiprocessing import cpu_count
import sys
sys.path.insert(0, '/scratch/akinbiyi')

from opPython.setupFolders import *

from simPython.makeSimPedFiles import *
from simPython.genSimZScores import *
from dataPrepPython.genZScores import *
from dataPrepPython.genLZCorr import *

from statsPython.setupELL import *
from statsPython.genELL import*
from statsPython.makeELLMCPVals import*
from statsPython.makeELLMarkovPVals import*
from statsPython.makeGBJPVals import *

from plotPython.plotPower import *
from genPython.makePSD import *

from datetime import datetime
import shutil
import subprocess
import psutil
from distutils.dir_util import copy_tree

warnings.simplefilter("error")

local=os.getcwd()+'/'

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
SnpSize=[40,40,40]
traitChr=[18,16,19]
snpChr=[snp for snp in range(1,len(SnpSize)+1)]
muEpsRange=[[0,0]]

parms={
    'etaGRM':.5,
    'etaError':.5,
    'ellBetaPpfEps':1e-12,
    'ellKRanLowEps':.1,
    'ellKRanHighEps':1.9,
    'ellKRanNMult':.4,
    'ellDSet':ellDSet,
    'minELLDecForInverse':4,
    'binsPerIndex':700,
    'local':local,
    'name':'testType1/',
    'numCores':cpu_count(),
    'numPCs':10,
    'snpChr':snpChr,
    'traitChr':traitChr,
    'numDecScore':3,
    'SnpSize':SnpSize,
    'transOnly':False,
    'colors':colors,
    'RefReps':1000000,    
    'maxSnpGen':5000,
    'simLearnType':'Full',
    'response':'hipRaw',
    'quantNormalizeExpr':False,
    'verbose':True,
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':muEpsRange,
    'traitSubset':list(range(100)),
    'fastlmm':True,
    'fastGrm':False,
    'eyeTrait':True
}

setupFolders(parms,'testType1')

for nInd in range(len(traitChr)):    
    print('makeSimPedFiles')
    makeSimPedFiles(parms)

    print('genZScores')
    DBCreateFolder('score',parms)
    DBCreateFolder('holds',parms)
    genZScores(parms)

    subprocess.call(['rm','-rf','scoreInit'])
    subprocess.call(['mv','score','scoreInit'])
    
    traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)

    parms['snpChr']=range(3,3+len(parms['muEpsRange'])+1)
    parms['traitChr']=traitChr[0:nInd+1]

    N=sum(traitData['chr'].isin(parms['traitChr']))
    parms['muEpsRange']=[[.25,int(np.round(N*.008))]]
    
    subprocess.call(['rm','-rf','score'])
    subprocess.call(['cp','-r','scoreInit','score'])
    
    DBCreateFolder('holds',parms)
    genSimZScores(parms)   
    
    print(nInd,2)

    DBCreateFolder('holds',parms)
    DBCreateFolder('diagnostics',parms)        
    DBCreateFolder('stats',parms)
    DBCreateFolder('ELL',parms)
    DBCreateFolder('pvals',parms)
    DBCreateFolder('LZCorr',parms)

    print(nInd,2,'genLZCorr')
    genLZCorr({**parms,'snpChr':2})

    print(nInd,2,'setupELL')
    t0,amt,pct=time.time(),np.sum(np.array([proc.memory_info().rss for proc in psutil.Process().children(recursive=True)]+
                        [psutil.Process().memory_info().rss])),np.sum(np.array([proc.memory_info().memory_percent() for proc in 
                        psutil.Process().children(recursive=True)]+[psutil.Process().memory_info().memory_percent()]))    
    setupELL(parms)
    t1,newAmt,newPct=time.time(),np.sum(np.array([proc.memory_info().rss for proc in psutil.Process().children(recursive=True)]+
                        [psutil.Process().memory_info().rss])),np.sum(np.array([proc.memory_info().memory_percent() for proc in 
                        psutil.Process().children(recursive=True)]+[psutil.Process().memory_info().memory_percent()]))    
    print(t1-t0,newAmt-amt,newPct-pct,flush=True)
    
    print(nInd,2,'genELL')
    genELL(parms)

    print(nInd,2,'makeELLMCPVals')
    makeELLMCPVals(parms)

    print(nInd,2,'makeELLMarkovPVals')
    makeELLMarkovPVals(parms)

    LZCorr=np.loadtxt('LZCorr/LZCorr',delimiter='\t')   
    offDiag=np.matmul(LZCorr,LZCorr.T)[np.triu_indices(N,1)].flatten()

    stat=ell(offDiag,N,np.array(ellDSet)*N)
    x.fit(N*70,N*500,1e-3,999e-3,500)
    
    cpFolder=str(N)+'/'
    DBCreateFolder(cpFolder,parms)
    subprocess.call(['cp','-r','pvals',cpFolder])
    subprocess.call(['cp','-r','stats',cpFolder])
    subprocess.call(['cp','-r','ELL',cpFolder])
    subprocess.call(['cp','-r','LZCorr',cpFolder])
    
    print(nInd,'plotPower')
    DBCreateFolder('diagnostics',parms)
    plotPower(parms)
    subprocess.call(['rm','-rf','diagnostics-'+str(N)])
    subprocess.call(['mv','diagnostics','diagnostics-'+str(N)])
    
