import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb
import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from dill.source import getsource
from opPython.setupFolders import *
from simPython.makeSimInputFiles import *
from dataPrepPython.genZScores import *
from multiprocessing import cpu_count
from plotPython.plotCorr import *
from plotPython.plotPower import *
from plotPython.plotZ import *

from scipy.stats import norm
import ELL.ell

def myMain(mainDef):
    colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
    ellDSet=[.1,.5]
    
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
        'muEpsRange':[],
        'maxSnpGen':5000,
        'transOnly':False,
        'YSeed':0,
        'snpsSeed':0,
        'logSource':True,
        'local':local
    }
    
    #['etaSq','numSubjects','numTraits','numSnps']
    #['realSnps','pedigreeSnps','randSnps','iidSnps','grmSnps','indepTraits','depTraits','quantNorm','stdNorm','noNorm']
    #['indepTraits','depTraits']
    #['gemma','fast','limix','lmm','lm','ped','bimbam','bed']
    #['gemmaStd','gemmaCentral','fast','limix','bed','bimbam','ped']
    ctrl={
        'parms':[0,1000,300,[2000,500]],
        'sim':['indepTraits','grmSnps','noNorm'],
        'ell':'indepTraits',
        'reg':['gemma','lmm','bimbam'],
        'grm':['gemma','std']
    }
    parms=setupFolders(ctrl,ops)
    numSnps=ctrl['parms'][-1]

    DBCreateFolder('grm',parms)
    DBCreateFolder('inputs',parms)
    makeSimInputFiles(parms)
    
    #######################################################################################################

    DBCreateFolder('score',parms)
    genZScores(parms,[len(numSnps)])
    
    #######################################################################################################

    z=np.loadtxt('score/waldStat-'+str(len(numSnps)),delimiter='\t')
    eta=np.loadtxt('score/eta-'+str(len(numSnps)),delimiter='\t')[0]
    Y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2:]
    zRef=norm.rvs(size=[int(parms['parms'][-1][-1]),int(parms['parms'][2])])
    
    #######################################################################################################

    DBCreateFolder('diagnostics',parms)
    
    plotZ(z,prefix='z-')
    plotZ(Y,prefix='y-')
    plotCorr(np.loadtxt('grm/gemma-1',delimiter='\t'),'grm')

    fig,axs = plt.subplots(1,1,dpi=50)   
    fig.set_figwidth(10,forward=True)
    fig.set_figheight(10,forward=True)
    axs.hist(np.log(eta),bins=30)
    axs.set_title('eta')
    axs.set_xlabel('eta')
    if parms['parms'][0]>0:
        axs.axvline(x=np.log(parms['parms'][0]),color='k')
    fig.savefig('diagnostics/eta.png')
    
    #######################################################################################################
    
    maxD=int(np.max(ellDSet)*parms['parms'][2])
    numTraits=parms['parms'][2]
    ell=np.empty(shape=[parms['parms'][-1][-1],maxD])
    pvals=np.sort(2*norm.sf(np.abs(z)))
    for i in range(maxD):
        ell[:,i]=beta.cdf(pvals[:,i],i+1,numTraits-i)
    ell=np.min(ell,axis=1)
    
    ellRef=np.empty(shape=[parms['parms'][-1][-1],int(np.max(ellDSet)*parms['parms'][2])])
    pvals=np.sort(2*norm.sf(np.abs(zRef)))
    for i in range(maxD):
        ellRef[:,i]=beta.cdf(pvals[:,i],i+1,numTraits-i)
    ellRef=np.sort(np.min(ellRef,axis=1))
        
    monteCarlo=np.empty(shape=len(ell))
    sortOrd=np.argsort(ell,axis=0)
    monteCarlo[sortOrd]=(1+np.searchsorted(ellRef,ell[sortOrd]))/(len(ellRef)+1)
    
    '''
    stat=ELL.ell.ell(np.array([.1,.5]),parms['parms'][2])
    monteCarlo=stat.monteCarlo(ellRef,ell)
    if 'depTraits' in parms['ell']:
        offDiag=np.corrcoef(z,rowvar=False)[np.triu_indices(parms['parms'][2],1)]
    if 'indepTraits' in parms['ell']:
        offDiag=np.array([0]*int(parms['parms'][2]*(parms['parms'][2]-1)/2))
        
    stat.fit(10*parms['parms'][2],1000*parms['parms'][2],3000,1e-6,offDiag) # numLamSteps0,numLamSteps1,numEllSteps,minEll

    #######################################################################################################
    
    refELL=stat.score(zRef)
    score=stat.score(z)

    #######################################################################################################

    monteCarlo=stat.monteCarlo(refELL,score)

    #######################################################################################################

    markov=stat.markov(refELL)

    #######################################################################################################
    '''
    cross=plotPower(monteCarlo,parms,'mc',['mc'])
    DBFinish(local,mainDef)
    #plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])
    

myMain(getsource(myMain))