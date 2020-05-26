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
from statsPython.psi import *
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

def plots(wald,vZ,psi,offDiag,refReps,maxRefReps,numCores,title):
    func=partial(ELL,psi)
    storey=partial(storeyQ,int(vZ.shape[1]*.5))

    plotPower(monteCarlo(cpma,wald,vZ,refReps,maxRefReps,numCores,'cpma'),'cpma-'+title)
    plotPower(monteCarlo(func,wald,vZ,refReps,maxRefReps,numCores,'ell'),'ell-'+title) 
    plotPower(markov(func,wald,psi,offDiag,numCores),'ellMarkov-'+title)  
    plotPower(monteCarlo(scoreTest,wald,vZ,refReps,maxRefReps,numCores,'scoreTest'),'scoreTest-'+title)      
    plotPower(monteCarlo(storey,wald,vZ,refReps,maxRefReps,numCores,'storeyQ'),'storeyQ-'+title)      
    plotPower(monteCarlo(minP,wald,vZ,refReps,maxRefReps,numCores,'minP'),'minP-'+title)     

    return()

def myMain(parms,fitH0,fitH1,plotH1):
    numH0Snps=parms['numH0Snps']
    numH1Snps=parms['numH1Snps']
    numKSnps=parms['numKSnps']
    numTraits=parms['numTraits']
    numSubjects=parms['numSubjects']
    pedigreeMult=parms['pedigreeMult']
    d=int(parms['d']*numTraits)
    rho=parms['rho']
    maxEta=parms['maxEta']
    minEta=parms['minEta']
    mu=parms['mu']
    n_assoc=parms['n_assoc']
    
    numCores=cpu_count()
    refReps=int(2e3)
    maxRefReps=int(1e2)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if fitH0:
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

        snpsH0=makePedigreeSnps(numSubjects,miceRange,numH0Snps,numCores)
        M=np.ones([numSubjects,1])

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        waldH0,etaH0=runLimix(Y,QS,np.ones([numSubjects,1]),snpsH0,0.9999)

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        snpsH1=makePedigreeSnps(numSubjects,miceRange,numH1Snps,numCores)
        waldH1,etaH1=runLimix(Y,QS,np.ones([numSubjects,1]),snpsH1,0.9999)

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        np.savetxt('Y',Y,delimiter='\t')
        np.savetxt('waldH0',waldH0,delimiter='\t')
        np.savetxt('waldH1',waldH1,delimiter='\t')
        np.savetxt('etaH0',etaH0,delimiter='\t')
        np.savetxt('etaH1',etaH1,delimiter='\t')
        np.savetxt('K',K,delimiter='\t')
        np.savetxt('M',M,delimiter='\t')
        np.savetxt('snpsH0',snpsH0,delimiter='\t')
        np.savetxt('snpsH1',snpsH1,delimiter='\t')
    else:
        waldH0=np.loadtxt('waldH0',delimiter='\t')
        waldH1=np.loadtxt('waldH1',delimiter='\t')
        etaH0=np.loadtxt('etaH0',delimiter='\t')
        etaH1=np.loadtxt('etaH1',delimiter='\t')
        Y=np.loadtxt('Y',delimiter='\t')
        K=np.loadtxt('K',delimiter='\t')        
        M=np.loadtxt('M',delimiter='\t');M=M.reshape(len(M),-1)        
        snpsH0=np.loadtxt('snpsH0',delimiter='\t')        
        snpsH1=np.loadtxt('snpsH1',delimiter='\t') 
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if fitH1:
        waldH100=runH1(16,100,waldH0,Y,K,M,snpsH0,etaH0)
        waldH10=runH1(16,10,waldH0,Y,K,M,snpsH0,etaH0)
        np.savetxt('waldH100',waldH100,delimiter='\t')
        np.savetxt('waldH10',waldH10,delimiter='\t')
    else:
        waldH100=np.loadtxt('waldH100',delimiter='\t')
        waldH10=np.loadtxt('waldH10',delimiter='\t')

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    vZ=np.corrcoef(waldH0,rowvar=False)
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    psiDF=psi(d,vZ,numLam=1e3,minEta=1e-9,numCores=16).compute()

    plots(waldH0,vZ,psiDF,offDiag,refReps,maxRefReps,numCores,'H0')
    
    if plotH1:
        plots(waldH100,vZ,psiDF,offDiag,refReps,maxRefReps,numCores,'H100')
        plots(waldH10,vZ,psiDF,offDiag,refReps,maxRefReps,numCores,'H10')

    #stat.plot(gbj('GBJ',wald,numCores=3,offDiag=offDiag),'gbj')
    #plotPower(gbj('GHC',wald,numCores=3,offDiag=offDiag),'ghc')
    

ops={
    'seed':323,
    'numKSnps':1000,
    'd':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':800,
    'numH0Snps':600,
    'numH1Snps':20,
    'numTraits':300,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'mu':16,
    'n_assoc':100,
    'rho':1,
    'maxEta':0.8,
    'minEta':0
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

myMain(parms,fitH0=True,fitH1=False,plotH1=False)

git('{} mice, {} snps, {} traits, subsample {}, rho {}'.format(parms['numSubjects'],parms['numH0Snps'],
    parms['numTraits'],parms['pedigreeMult'],parms['rho']))
