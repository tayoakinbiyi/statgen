import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb
import sys

sys.path=['source']+sys.path

from opPython.setupFolders import *

from simPython.makeSimInputFiles import *
from dataPrepPython.genZScores import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from plotPython.plotZ import *

from scipy.stats import norm

from ELL.ell import *

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]

#['gemma','fast','lmm','lm','bed','bimbam','ped','gemmaStdGrm','gemmaCentralGrm','fastGrm']
#['quantNorm','stdNorm',etaSq,numSubjects,numTraits,numSnps]
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
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
}

#######################################################################################################

for exp in [['gemma','bimbam'],['gemma','bed'],['fast','bed'],['fast','ped']]:
    ctrl={
        'sim':['indepTraits','randSnps',0.5,200,300,[5000,500]],
        'model':'indepTraits',
        'data':[exp[0],'lmm',exp[1],'gemmaStdGrm']
    }
    parms=setupFolders(ctrl,ops)

    DBCreateFolder('grm',parms)
    DBCreateFolder('inputs',parms)
    makeSimInputFiles(parms)

    #######################################################################################################

    DBCreateFolder('score',parms)
    genZScores(parms,[len([ctrl['sim'][-1]])])

    #######################################################################################################

    z=np.loadtxt('score/waldStat-'+str(len([ctrl['sim'][-1]])),delimiter='\t')

    #######################################################################################################

    DBCreateFolder('diagnostics',parms)
    plotZ(z)

    DBFinish(parms)
    
#######################################################################################################
'''
offDiag=np.array([0]*int(numTraits*(numTraits-1)/2))
stat=ell(np.array(ellDSet),offDiag)

#######################################################################################################

#stat.load()
stat.fit(10*numTraits,1000*numTraits,3000,1e-6) # numLamSteps0,numLamSteps1,numEllSteps,minEll
stat.save()

#######################################################################################################

ell=stat.score(z)
ref=stat.score(norm.rvs(size=[int(parms['refReps']),numTraits]))

#######################################################################################################

monteCarlo=stat.monteCarlo(ref,ell)

#######################################################################################################

markov=stat.markov(ell)

#######################################################################################################

plotPower(monteCarlo,parms,'mc',['mc-'+str(x) for x in ellDSet])
plotPower(markov,parms,'markov',['markov-'+str(x) for x in ellDSet])
#pd.DataFrame(monteCarlo,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/monteCarlo.csv',index=False)
#pd.DataFrame(markov,columns=ellDSet).quantile([.05,.01],axis=0).to_csv('diagnostics/markov.csv',index=False)
'''

