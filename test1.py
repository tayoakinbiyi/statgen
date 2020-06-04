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
    mcMethodNames=parms['mcMethodNames']
    markovMethodNames=parms['markovMethodNames']
    numHermites=parms['numHermites']
    
    numCores=parms['numCores']
    refReps=parms['refReps']
    maxRefReps=parms['maxRefReps']
    numLam=parms['numLam']
    minEta=parms['minEta']
    mu=parms['mu']
    
    eps=parms['eps']
    maxIter=parms['maxIter']
            
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    miceRange=np.random.choice(208,int(pedigreeMult*208),replace=False)    

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'makeY' in fit:
        K=linear_kinship(makePedigreeSnps(numSubjects,miceRange,numKSnps,numCores),verbose=True)    
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
        
        np.savetxt('Y',Y,delimiter='\t')
        np.savetxt('K',K,delimiter='\t')
        np.savetxt('M',M,delimiter='\t')

    if 'loadY' in fit:
        Y=np.loadtxt('Y',delimiter='\t')
        K=np.loadtxt('K',delimiter='\t')        
        M=np.loadtxt('M',delimiter='\t');M=M.reshape(len(M),-1)        
        QS=economic_qs(K)

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitH0' in fit:
        snpsH0=makePedigreeSnps(numSubjects,miceRange,numH0Snps,numCores)
        waldH0,etaH0=runLimix(Y,QS,np.ones([numSubjects,1]),snpsH0,0.9999)

        np.savetxt('waldH0',waldH0,delimiter='\t')
        np.savetxt('etaH0',etaH0,delimiter='\t')
        np.savetxt('snpsH0',snpsH0,delimiter='\t')

    if 'loadH0' in fit:
        waldH0=np.loadtxt('waldH0',delimiter='\t')
        etaH0=np.loadtxt('etaH0',delimiter='\t')
        snpsH0=np.loadtxt('snpsH0',delimiter='\t')        
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitH1' in fit:
        snpsH1=makePedigreeSnps(numSubjects,miceRange,numH1Snps,numCores)
        waldH1,etaH1=runLimix(Y,QS,np.ones([numSubjects,1]),snpsH1,0.9999)

        np.savetxt('waldH1',waldH1,delimiter='\t')
        np.savetxt('etaH1',etaH1,delimiter='\t')
        np.savetxt('snpsH1',snpsH1,delimiter='\t')
            
    if 'loadH1' in fit:
        waldH1=np.loadtxt('waldH1',delimiter='\t')
        etaH1=np.loadtxt('etaH1',delimiter='\t')
        snpsH1=np.loadtxt('snpsH1',delimiter='\t') 

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    ds=[]
    if 'fitPower' in fit:
        for n in n_assoc:
            ds+=[runH1(mu,n,waldH1,Y,K,M,snpsH1,etaH1)]
            np.savetxt('waldH'+str(n),ds[-1],delimiter='\t')
            
    if 'loadPower' in fit:
        for n in n_assoc:
            ds+=[np.loadtxt('waldH'+str(n),delimiter='\t')]

    dsNames=['H'+str(x) for x in n_assoc]
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    vZ=np.corrcoef(waldH0,rowvar=False)
    offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitPsi' in fit:
        psiDF=psi(calD,vZ,numLam,minEta,numCores,eps,maxIter,numHermites).compute()
        np.savetxt('psi',psiDF,delimiter='\t')
    elif 'loadPsi' in fit:
        psiDF=np.loadtxt('psi',delimiter='\t',dtype=[('lam','float64'),('eta','float64')])
    else:
        psiDF=None
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    mcFuncs={
        'ELL':partial(ELL,psiDF,numCores),
        'cpma':partial(cpma,numCores),
        'score':partial(score,numCores),
        'storey':partial(storeyQ,numCores),
        'minP':partial(minP,numCores)
    }
    
    markovFuncs={
        'GBJ':partial(gbjLoop,'GBJ'),
        'GHC':partial(gbjLoop,'GHC'),
        'markov':partial(markovLoop,psiDF)
    }

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitRef' in fit:
        ref={key:genRef(f,refReps,maxRefReps,vZ,key) for key,f in mcFuncs.items() if key in mcMethodNames}
        for key in mcMethodNames:
            np.savetxt('ref-'+key,ref[key],delimiter='\t')
    
    if 'loadRef' in fit:
        ref={key:np.loadtxt('ref-'+key,delimiter='\t') for key in mcMethodNames}
            
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    pvalFuncs={}
    for key in mcMethodNames:
        pvalFuncs[key]=partial(mc,mcFuncs[key],vZ,ref[key],key)
    for key in markovMethodNames:
        pvalFuncs[key]=partial(markovFuncs[key],vZ,numCores)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    methods={x[0]:x[1] for f,nm in [(pvalFuncs.items(),mcMethodNames+markovMethodNames)] for x in f if x[0] in nm}
    if 'plotType1' in fit:
        plots([-np.sort(-np.abs(waldH0)) for x in ds],['H0'],methods)
        
    if 'plotPower' in fit:
        plots([-np.sort(-np.abs(x)) for x in ds],dsNames,methods)
    
ops={
    'seed':None,
    'numKSnps':10000,
    'calD':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':1200,
    'numH0Snps':10000,
    'numH1Snps':200,
    'numTraits':1200,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'rho':1,
    'refReps':int(1e6),
    'maxRefReps':int(1e5),
    'minEta':1e-10,
    'numLam':1e3,
    'mu':2.8,
    'eps':1e-11,
    'maxIter':1e2,
    'numHermites':150,
    'numCores':cpu_count(),
    'fit':['loadH0','loadH1','plotPower','loadPsi','loadY','fitPower','loadRef'],
    'n_assoc':[10],#,30,50,70,80,100,150],
    'mcMethodNames':['ELL'],#,'cpma','score','storey','minP'],
    'markovMethodNames':['markov']
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

createDiagnostics(parms['seed'])
log(parms)

myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, rho {}'.format(parms['numSubjects'],parms['numH0Snps'],
    parms['numTraits'],parms['pedigreeMult'],parms['rho']))
