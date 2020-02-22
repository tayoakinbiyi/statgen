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

snpSize=[500]
numSubjects=600
numTraits=400
etaSq=0

ellDSet=[.1,.5]
colors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(0,.5,0),(.5,0,0),(0,0,.5)]
traitChr=[18]#,20,16,19]
snpChr=[snp for snp in range(1,len(snpSize)+1)]

#['gemmaLmm','fastLmm','lmm','lm','bed','bimbam','ped','gemmaStdGrm','gemmaNoStdGrm','fastGrm']

ctrl={
    'etaSq':0,
    'sim':['indepTraits','randSnps',etaSq,numSubjects,numTraits,snpSize],#['simDep','real','simIndep']
    'model':'indepTraits',#['indep','dep']
    'data':['gemma','lmm','bimbam','gemmaStdGrm'], 
}
ops={
    'file':sys.argv[0],
    'ellDSet':ellDSet,
    'numCores':cpu_count(),
    'snpChr':snpChr,
    'traitChr':traitChr,
    'colors':colors,
    'refReps':1e6,    
    'simLearnType':'Full',
    'response':'hipRaw',
    'numSnpChr':18,
    'numTraitChr':21,
    'muEpsRange':[],
    'maxSnpGen':5000,
    'transOnly':False,
    'numTraits':numTraits
}

#######################################################################################################

parms=setupFolders(ctrl,ops)

DBCreateFolder('diagnostics',parms)
DBCreateFolder('inputs',parms)
DBCreateFolder('score',parms)
DBCreateFolder('grm',parms)

makeSimInputFiles(parms)

genZScores(parms)

#######################################################################################################

z=np.loadtxt('score/waldStat-1',delimiter='\t')
print(np.min(np.mean(z**2,axis=0)),np.max(np.mean(z**2,axis=0)))
plotZ(z)

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

DBFinish(parms)