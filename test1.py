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
from statsPython.score import *
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

def myMain(parms):
    numH0Snps=parms['numH0Snps']
    numH1Snps=parms['numH1Snps']
    numKSnps=parms['numKSnps']
    numTraits=parms['numTraits']
    numSubjects=parms['numSubjects']
    pedigreeMult=parms['pedigreeMult']
    calD=int(parms['calD']*numTraits)
    rho=parms['rho']
    fit=parms['fit']
    n_assoc=parms['n_assoc']
    methodNames=parms['methodNames']
    
    numCores=parms['numCores']
    refReps=parms['refReps']
    maxRefReps=parms['maxRefReps']
    numLam=parms['numLam']
    minEta=parms['minEta']
    mu=parms['mu']
    
    eps=parms['eps']
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitH0' in fit:
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
        # eta is beta(1,10)
        eta=np.diag(beta.rvs(1,10,size=[numTraits]))
        C_u=eta**0.5@(rho*np.corrcoef(ailY[0:104],rowvar=False)+(1-rho)*np.eye(numTraits))@eta**0.5
        C_e=(np.eye(numTraits)-eta)**0.5@(rho*np.corrcoef(ailY[104:],rowvar=False)+
            (1-rho)*np.eye(numTraits))@(np.eye(numTraits)-eta)**0.5

        LC_u=makeL(C_u)
        LC_e=makeL(C_e)

        Y=LK@norm.rvs(size=[numSubjects,numTraits])@LC_u.T+norm.rvs(size=[numSubjects,numTraits])@LC_e.T
        M=np.ones([numSubjects,1])

        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        snpsH0=makePedigreeSnps(numSubjects,miceRange,numH0Snps,numCores)
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
    if 'loadH0' in fit:
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

    ds=[waldH0]
    if 'fitH1' in fit:
        for n in n_assoc:
            ds+=[runH1(mu,n,waldH1,Y,K,M,snpsH1,etaH1)]
            np.savetxt('waldH'+str(n),ds[-1],delimiter='\t')
    if 'loadH1' in fit:
        for n in n_assoc:
            ds+=[np.loadtxt('waldH'+str(n),delimiter='\t')]

    dsNames=['H0']+['H'+str(x) for x in n_assoc]
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    vZ=np.corrcoef(waldH0,rowvar=False)
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitPsi' in fit:
        psiDF=psi(calD,vZ,numLam,minEta,numCores,eps).compute()
        np.savetxt('psi',psiDF,delimiter='\t')
    elif 'loadPsi' in fit:
        psiDF=np.loadtxt('psi',delimiter='\t',dtype=[('lam','float64'),('eta','float64')])
    else:
        psiDF=None

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    funcs={
        'ELL':partial(mc,partial(ELL,psiDF,numCores),'ELL',refReps,maxRefReps,vZ), 
        'cpma':partial(mc,partial(cpma,numCores),'cpma',refReps,maxRefReps,vZ), 
        'score':partial(mc,partial(score,numCores),'score',refReps,maxRefReps,vZ), 
        'storey':partial(mc,partial(storeyQ,numCores),'storeyQ',refReps,maxRefReps,vZ), 
        'minP':partial(mc,partial(minP,numCores),'minP',refReps,maxRefReps,vZ), 
        'markov':partial(markovLoop,psiDF,vZ,numCores),
        'GBJ':partial(gbjLoop,'GBJ',numCores,vZ),
        'GHC':partial(gbjLoop,'GHC',numCores,vZ)
    }
    
    methods={x[0]:x[1] for f,nm in [(funcs.items(),methodNames)] for x in f if x[0] in nm}
    plots([-np.sort(-np.abs(x)) for x in ds],dsNames,methods)
    
ops={
    'seed':None,
    'numKSnps':100,
    'calD':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':300,
    'numH0Snps':100,
    'numH1Snps':100,
    'numTraits':300,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'rho':1,
    'refReps':int(1e3),
    'maxRefReps':int(1e2),
    'minEta':1e-10,
    'numLam':5e3,
    'mu':5,
    'eps':1e-10,
    'numCores':cpu_count(),
    'fit':['loadH0','loadH1','fitPsi'],
    'n_assoc':[10],#,30,50,70,80,100,150],
    'methodNames':['score']#['ELL','cpma','score','storey','minP','markov']
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, rho {}'.format(parms['numSubjects'],parms['numH0Snps'],
    parms['numTraits'],parms['pedigreeMult'],parms['rho']))
