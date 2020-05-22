import matplotlib
matplotlib.use('agg')
import numpy as np
import pdb

import sys
import os

sys.path=[os.getcwd()+'/source']+sys.path
from dill.source import getsource
from dataPrepPython.runLimix import *
from multiprocessing import cpu_count
from plotPython.plotPower import *
from utility import *
from plotPython.myHist import *
from plotPython.plotPower import *
from limix.model.lmm import LMM
from limix.stats import linear_kinship
from dataPrepPython.makePedigreeSnps import *
from numpy_sugar.linalg import economic_qs
from statsPython.scoreTest import *
from statsPython.storeyQ import *
from statsPython.minP import *
from statsPython.preComputeELL import *
from dataPrepPython.runH1 import *
from statsPython.ELL import *
from statsPython.gbj import *
from statsPython.cpma import *
from statsPython.monteCarlo import *
from statsPython.markov import *
from functools import partial
from scipy.stats import norm, uniform
from plotPython.plotCorr import *

from rpy2.robjects.packages import importr

def plots(wald,vZ,lamEllByK,ellGrid,offDiag,refReps,maxRefReps,numCores,title):
    func=partial(ELL,lamEllByK,ellGrid)
    storey=partial(storeyQ,int(vZ.shape[1]*.5))

    plotPower(monteCarlo(cpma,wald,vZ,refReps,maxRefReps,numCores,'cpma'),'cpma-'+title)
    plotPower(monteCarlo(func,wald,vZ,refReps,maxRefReps,numCores,'ell'),'ell-'+title)      
    plotPower(markov(func,wald,lamEllByK,ellGrid,offDiag,numCores),'ellMarkov-'+title)  
    plotPower(monteCarlo(scoreTest,wald,vZ,refReps,maxRefReps,numCores,'scoreTest'),'scoreTest-'+title)      
    plotPower(monteCarlo(storey,wald,vZ,refReps,maxRefReps,numCores,'storeyQ'),'storeyQ-'+title)      
    plotPower(monteCarlo(minP,wald,vZ,refReps,maxRefReps,numCores,'minP'),'minP-'+title)     

    return()

def myMain(parms,fit):
    numDataSnps=parms['numDataSnps']
    numKSnps=parms['numKSnps']
    numTraits=parms['numTraits']
    numSubjects=parms['numSubjects']
    pedigreeMult=parms['pedigreeMult']
    d=int(parms['d']*numTraits)
    rho=parms['rho']
    maxEta=parms['maxEta']
    minEta=parms['minEta']
    
    numCores=cpu_count()
    refReps=int(2e6)
    maxRefReps=int(1e5)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if fit:
        miceRange=np.random.choice(208,int(pedigreeMult*208),replace=False)    

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        K=linear_kinship(makePedigreeSnps(numSubjects,miceRange,numKSnps,numCores),verbose=True)    
        QS=economic_qs(K)
        LK=makeL(K)

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        ailY=pd.read_csv('../data/hipRaw.txt',sep='\t',index_col=0,header=0).values[:,0:numTraits]
        eta=np.diag(minEta+(maxEta-minEta)*uniform.rvs(size=[numTraits]))
        C_u=eta**0.5@(rho*np.corrcoef(ailY[0:104],rowvar=False)+(1-rho)*np.eye(numTraits))@eta**0.5
        C_e=(np.eye(numTraits)-eta)**0.5@(rho*np.corrcoef(ailY[104:],rowvar=False)+
            (1-rho)*np.eye(numTraits))@(np.eye(numTraits)-eta)**0.5

        LC_u=makeL(C_u)
        LC_e=makeL(C_e)

        Y=LK@norm.rvs(size=[numSubjects,numTraits])@LC_u.T+norm.rvs(size=[numSubjects,numTraits])@LC_e.T

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        snps=makePedigreeSnps(numSubjects,miceRange,numDataSnps,numCores)
        M=np.ones([numSubjects,1])

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        wald,eta=runLimix(Y,QS,np.ones([numSubjects,1]),snps,0.9999)

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        np.savetxt('Y',Y,delimiter='\t')
        np.savetxt('wald',wald,delimiter='\t')
        np.savetxt('eta',eta,delimiter='\t')
        np.savetxt('K',K,delimiter='\t')
        np.savetxt('M',M,delimiter='\t')
        np.savetxt('snps',snps,delimiter='\t')
    else:
        wald=np.loadtxt('wald',delimiter='\t')
        eta=np.loadtxt('eta',delimiter='\t')
        Y=np.loadtxt('Y',delimiter='\t')
        K=np.loadtxt('K',delimiter='\t')        
        M=np.loadtxt('M',delimiter='\t')        
        snps=np.loadtxt('snps',delimiter='\t')        
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    vZ=np.corrcoef(wald,rowvar=False)
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    lamEllByK,ellGrid=preComputeELL(d,vZ,numCores=16).preCompute(1e4,1e-9)
    
    waldH1=runH1(0.2,5,wald,Y,K,M,snps,eta)

    plots(wald,vZ,lamEllByK,ellGrid,offDiag,refReps,maxRefReps,numCores,'H0')
    plots(waldH1,vZ,lamEllByK,ellGrid,offDiag,refReps,maxRefReps,numCores,'H1')

    #stat.plot(gbj('GBJ',wald,numCores=3,offDiag=offDiag),'gbj')
    #plotPower(gbj('GHC',wald,numCores=3,offDiag=offDiag),'ghc')
    

ops={
    'seed':1023,
    'numKSnps':500,
    'd':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':300,
    'numDataSnps':200,
    'numPowerSnps':100,
    'numTraits':50,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'mu':1,
    'n_assoc':1,
    'rho':1,
    'maxEta':0.8,
    'minEta':0
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

myMain(parms,True)

git('{} mice, {} snps, {} traits, subsample {}, rho {}'.format(parms['numSubjects'],parms['numDataSnps'],
    parms['numTraits'],parms['pedigreeMult'],parms['rho']))
