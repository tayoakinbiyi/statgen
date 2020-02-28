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
        'YSeed':973243,
        'snpsSeed':23425,
        'logSource':True,
        'local':local
    }
    
    count=0
    
    parms=setupFolders({},ops)
    z=[]
    
    ref=pd.DataFrame()
    for exp in [{'count':count,'soft':soft,'eta':eta,'fmt':fmt,'numSubjects':200,'cross':False} for 
            soft in ['gemma','fast'] for eta in [0,.2,.5] for fmt in ['ped','bed','bimbam']  
            if not ((fmt=='ped' and soft=='gemma') or (fmt=='bimbam' and soft=='fast'))]:
        os.chdir(local+ops['file'][:-3])
        ctrl={
            'count':count,
            'parms':[exp['eta'],200,100,[1000,300]],
            'sim':['indepTraits','grmSnps','noNorm'],
            'ell':'indepTraits',
            'reg':[exp['soft'],'lmm',ex['fmt']],
            'grm':['fast','std']
        }
        parms={**ctrl,**ops}
        numSnps=ctrl['parms'][-1]

        DBLog(ctrl)

        subprocess.call(['rm','-f','log'])
        DBLog(json.dumps(ctrl,indent=3))

        DBCreateFolder('grm',parms)
        DBCreateFolder('inputs',parms)
        makeSimInputFiles(parms)

        #######################################################################################################

        DBCreateFolder('score',parms)
        genZScores(parms,list(range(len(numSnps)-1,len(numSnps)+1)))
        z=np.loadtxt('score/waldStat-'+str(len(numSnps)),delimiter='\t')
        eta=np.loadtxt('score/eta-'+str(len(numSnps)),delimiter='\t')[0]
        Y=np.loadtxt('inputs/Y.phe',delimiter='\t')[:,2:]

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
        axs.axvline(x=np.log(parms['parms'][0]),color='k')
        fig.savefig('diagnostics/eta.png')
        
        ref=ref.append(exp,ignore_index=True)
        
        count+=1
        
    np.savetxt('score/z',np.concatenate(z,axis=0),delimiter='\t')

    ref.to_csv('diagnosticsAll/ref.tsv',delimiter='\t',index=False)

myMain(getsource(myMain))