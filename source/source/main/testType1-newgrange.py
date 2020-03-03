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
    stat=ELL.ell.ell(np.array([.1,.5]),numTraits)
    stat.fit(10*numTraits,1000*numTraits,3000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
    refELL=stat.score(norm.rvs(size=[int(1e6),numTraits]))
        
    ref=pd.DataFrame()
    for exp in [{'count':count,'soft':soft,'eta':eta,'fmt':fmt,'numSubjects':numSubjects,'traits':traits,'cross':False} for 
            soft in ['gemma','fast'] for traits in ['indepTraits'] for eta in 
            [0,.33,.66] for fmt in ['ped','bed','bimbam'] for numSubjects in [400,2000,10000] 
            if not ((fmt=='ped' and soft=='gemma') or (fmt=='bimbam' and soft=='fast'))]:
        os.chdir(local+ops['file'][:-3])
        np.random.seed(27)
        ctrl={
            'count':count,
            'parms':[exp['eta'],exp['numSubjects'],numTraits,[5000,500]],
            'sim':[exp['traits'],'pedigreeSnps','noNorm'],
            'ell':exp['traits'],
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

        z=np.loadtxt('score/waldStat-'+str(len(numSnps)),delimiter='\t')
        eta=np.loadtxt('score/eta-'+str(len(numSnps)),delimiter='\t')

        #######################################################################################################

        DBCreateFolder('diagnostics',parms)
        plotZ(z)
        
        fig,axs=plt.subplots(1,1)
        fig.set_figwidth(10,forward=True)
        fig.set_figheight(10,forward=True)
        axs.scatter(np.log(np.mean(eta,axis=0)),np.mean(z**2,axis=0))
        axs.set_title('traitMean z**2 ~ eta')
        axs.axvline(x=exp['eta'])
        axs.axhline(y=1)
        fig.savefig('diagnostics/zVsEta.png')
        
        #######################################################################################################

        #z=np.loadtxt('score/waldStat-2',delimiter='\t')
        #offDiag=np.corrcoef(z,rowvar=False)[np.triu_indices(numTraits,1)]
        #stat.fit(10*numTraits,1000*numTraits,3000,1e-6,offDiag) # numLamSteps0,numLamSteps1,numEllSteps,minEll

        #refELL=stat.score(norm.rvs(size=[int(1e6),numTraits]))

        #######################################################################################################

        score=stat.score(z)

        #######################################################################################################

        monteCarlo=stat.monteCarlo(refELL,score)

        #######################################################################################################

        markov=stat.markov(score)

        #######################################################################################################

        cross=plotPower(monteCarlo,parms,'mc',['mc-'+str(x) for x in ellDSet])
        plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])

        #######################################################################################################
        
        exp['cross']=cross[-1]
        ref=ref.append(exp,ignore_index=True)
        
        subprocess.call(['touch','diagnostics/'+str(count)])

        DBFinish(local,mainDef)
        count+=1

    ref.to_csv(local+parms['file']+'-ref.tsv',delimiter='\t',index=False)
    

myMain(getsource(myMain))