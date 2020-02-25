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
from plotPython.plotPower import *
from plotPython.plotZ import *

from scipy.stats import norm
import ELL.ell

def myMain(mainDef):
    colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
    ellDSet=[.1,.5]
    
    numSubjects=200
    numTraits=300

    #['etaSq','numSubjects','numTraits','numSnps']
    #['realSnps','pedigreeSnps','randSnps','indepTraits','depTraits','quantNorm','stdNorm','noNorm']
    #['indepTraits','depTraits']
    #['gemma','fast','lmm','lm','ped','bimbam','bed']
    #['gemmaStd','gemmaCentral','fast','bed','bimbam','ped']
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
    
    count=0
    
    parms=setupFolders({},ops)
    z=[]
    
    ref=pd.DataFrame()
    for exp in [{'count':count,'soft':soft,'eta':eta,'fmt':fmt,'numSubjects':400,'cross':False} for 
            soft in ['gemma','fast'] for eta in [0,.5] for fmt in ['ped','bed','bimbam']  
            if not ((fmt=='ped' and soft=='gemma') or (fmt=='bimbam' and soft=='fast'))]:
        os.chdir(local+ops['file'][:-3])
        np.random.seed(55923)
        ctrl={
            'count':count,
            'parms':[exp['eta'],exp['numSubjects'],numTraits,[5000,500]],
            'sim':['indepTraits','pedigreeSnps','noNorm'],
            'ell':'indepTraits',
            'reg':[exp['soft'],'lmm',exp['fmt']],
            'grm':['gemma','std']
        }
        parms={**ctrl,**ops}
        
        DBLog(ctrl)

        subprocess.call(['rm','-f','log'])
        DBLog(json.dumps(ctrl,indent=3))
        
        DBCreateFolder('grm',parms)
        DBCreateFolder('inputs',parms)
        makeSimInputFiles(parms)
        
        #######################################################################################################

        DBCreateFolder('score',parms)
        numSnps=ctrl['parms'][-1]
        genZScores(parms,list(range(len(numSnps)-1,len(numSnps)+1)))

        #######################################################################################################

        z+=[np.loadtxt('score/waldStat-2',delimiter='\t').reshape(1,-1)]
        
        ref=ref.append(exp,ignore_index=True)
        
        count+=1
        
    np.savetxt('score/z',np.concatenate(z,axis=0),delimiter='\t')

    ref.to_csv('diagnosticsAll/ref.tsv',delimiter='\t',index=False)

myMain(getsource(myMain))