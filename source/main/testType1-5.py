import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb
import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from dill.source import getsource
from opPython.setupFolders import *
from dataPrepPython.makeSim import *
from dataPrepPython.genZScores import *
from multiprocessing import cpu_count
from plotPython.plotCorr import *
from plotPython.plotPower import *
from plotPython.plotZ import *
from plotPython.myQQ import *
from genPython.makePSD import *

from statsPython.makeGBJPVals import *

from scipy.stats import norm
import ELL.ell
from statsPython.ellFull import *
from zipfile import ZipFile

def myMain(mainDef):
    colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
    ellDSet=np.array([.1,.5])
    
    local=os.getcwd()+'/'
    ops={
        'file':sys.argv[0],
        'numCores':cpu_count(),
        'colors':colors,
        'refReps':1e6,    
        'simLearnType':'Full',
        'response':'hipRaw',
        'numSnpChr':18,
        'numTraitChr':21,
        'maxSnpGen':5000,
        'transOnly':False,
        'ySeed':0,
        'snpSeed':0,
        'logSource':True
    }
    
    ctrl={
        'parms':[0,2000,300,[10000,1000]],
        'snpParm':['pedigreeSnps'],
        'yParm':['indepTraits','noNorm'],
        'ell':'depTraits',
        'grmParm':['limix'],
        'reg':['limix','lmm','bimbam']
    }
    diagnostics(mainDef,ctrl)    
    parms=setupFolders(ctrl,ops)
    numSnps=ctrl['parms'][3]
    numSubjects=ctrl['parms'][1]
    numTraits=ctrl['parms'][2]
    
    #######################################################################################################
    
    makeSim(parms)    
    genZScores(parms,[len(numSnps)])
    
    #######################################################################################################
    
    z=np.loadtxt('score/waldStat-'+str(len(numSnps)),delimiter='\t')
        
    #######################################################################################################

    if 'depTraits' in parms['ell']:
        corr=np.corrcoef(z,rowvar=False)
        offDiag=corr[np.triu_indices(numTraits,1)]
        L=makePSD(corr)
    if 'indepTraits' in parms['ell']:
        offDiag=np.array([0]*int(numTraits*(numTraits-1)/2))
        L=np.eye(numTraits)

    #######################################################################################################   
    
    stat=ELL.ell.ell(np.array([.1,.5]),numTraits)
    stat.fit(10,3000,4000,1e-7,offDiag) # numLamSteps0,numLamSteps1,numEllSteps,minEll
    
    zRef=np.matmul(norm.rvs(size=[int(ops['refReps']),numTraits]),L.T)
    refELL=stat.score(zRef)
    score=stat.score(z)

    mc=stat.monteCarlo(refELL,score)
    plotPower(mc,parms,'mc-log',['mc-'+str(x) for x in ellDSet],log=True)
    plotPower(mc,parms,'mc',['mc-'+str(x) for x in ellDSet],log=False)

    markov=stat.markov(score)
    plotPower(markov,parms,'markov-log',['markov-'+str(x) for x in ellDSet],log=True)
    plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet],log=False)
    
    full,ell=ellFull(parms,z,ellDSet,L)
    myHist(np.mean(ell,axis=0),'rowMeanEll')
    plotPower(full,parms,'full-log',['full-'+str(x) for x in ellDSet],log=True)
    plotPower(full,parms,'full',['full-'+str(x) for x in ellDSet],log=False)
    
    full,ell=ellFull(parms,z,ellDSet,np.eye(numTraits))
    myHist(np.mean(ell,axis=0),'rowMeanEll-I')
    plotPower(full,parms,'full-log-I',['full-I-'+str(x) for x in ellDSet],log=True)
    plotPower(full,parms,'full-I',['full-I-'+str(x) for x in ellDSet],log=False)

    #######################################################################################################
    '''
    Pgbj,Pghc,Phc,Pbj,PminP=makeGBJPVals(parms,z,offDiag)
        
    for obj in [[Pgbj,'Pgbj',[.5]],[Pghc,'Pghc',[.5]],[Phc,'Phc',[.5]],[Pbj,'Pbj',[.5]],[PminP,'PminP',[.5]]]:
        plotPower(obj[0],parms,obj[1]+'-log',[obj[1]+'-'+str(x) for x in obj[2]],log=True,myZip=myZip)
        plotPower(obj[0],parms,obj[1],[obj[1]+'-'+str(x) for x in obj[2]],log=False,myZip=myZip)
    '''
    #######################################################################################################
    
    plotZ(z,prefix='z-')
    
    git(local)
    

myMain(getsource(myMain))