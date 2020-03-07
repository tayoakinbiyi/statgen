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
        'ySeed':0,
        'snpSeed':0,
        'logSource':True,
        'local':local
    }
    
    #['etaSq','numSubjects','numTraits','numSnps']
    #['realSnps','pedigreeSnps','iidSnps','grmSnps','indepTraits','depTraits','quantNorm','stdNorm','noNorm']
    #['indepTraits','depTraits']
    #['gemma','fast','limix','lmm','lm','ped','bimbam','bed']
    #['gemmaStd','gemmaCentral','fast','limix','bed','bimbam','ped']
    ctrl={
        'parms':[0.3,800,300,[10000,1000]],
        'snpParm':['pedigreeSnps'],
        'yParm':['indepTraits','noNorm'],
        'ell':'indepTraits',
        'grmParm':['gemma']
    }
    parms=setupFolders(ctrl,ops)
    numSnps=ctrl['parms'][-1]
    makeSim(parms)

    #######################################################################################################
    
    DBCreateFolder('diagnostics',parms)
    DBLog(ctrl)
    
    z={}
    lmms=[['limix','bimbam'],['pylmm','bimbam'],['gemma','bimbam'],['fast','ped']]
    
    for lmm in lmms:
        parms['grmParm']=[lmm[0]]
        makeSim(parms,genSnps=False,genGrm=True,genY=False,genCov=False)
            
        parms['reg']=[lmm[0],'lmm',lmm[1]]
        genZScores(parms,[len(numSnps)])
        subprocess.call(['mv','score/waldStat-'+str(len(numSnps)),'score/'+lmm[0]+'-'+str(len(numSnps))])
        z[lmm[0]]=np.loadtxt('score/'+parms['reg'][0]+'-'+str(len(numSnps)),delimiter='\t')
        plotZ(z[lmm[0]],prefix=lmm[0]+'-')
        
    for lmm1 in range(0,len(lmms)-1):
        for lmm2 in range(lmm1+1,len(lmms)):
            myQQ(z[lmms[lmm1][0]].flatten(),z[lmms[lmm2][0]].flatten(),'lmm: '+lmms[lmm2][0]+' vs '+lmms[lmm1][0],
                 ylabel=lmms[lmm2][0],xlabel=lmms[lmm1][0])

    DBFinish(local,mainDef)
    #plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])
    

myMain(getsource(myMain))