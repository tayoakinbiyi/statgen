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

def plots(wald,vZ,psi,offDiag,refReps,maxRefReps,numCores,dataSetName):
    func=partial(ELL,psi)
    storey=partial(storeyQ,int(vZ.shape[1]*.5))
    
    funcs=[func,cpma,scoreTest,storey,minP]
    methName=['ELL','cpma','scoreTest','storey','minP','ellMarkov']

    pvalSet={nm:[] for nm in dataSetName}
    
    for fInd in range(len(funcs)):
        print('beginning genRef for {}'.format(methName[fInd]),flush=True)
        func=funcs[fInd]
        
        ref,mins=genRef(func,refReps,maxRefReps,vZ,numCores)
        log('{} : {} min'.format('genRef-'+methName[fInd],mins))
        
        for df in range(len(wald)):
            print('beginning score for {} {}'.format(methName[fInd],dataSetName[df]),flush=True)
            test,mins=func(wald[df],numCores)
            log('mc score {} : {} dataset {} min'.format(methName[fInd],dataSetName[df],mins))
    
            print('beginning mcPVal for {} {}'.format(methName[fInd],dataSetName[df]),flush=True)
            pval,mins=mcPVal(test,ref)
            log('mc pval {} : {} dataset {} min'.format(methName[fInd],dataSetName[df],mins))
            
            pvalSet[dataSetName[df]]+=[pval.reshape(-1,1)]
    
            plotPower(pval,methName[fInd]+'-'+dataSetName[df],cols=[methName[fInd]+'-'+dataSetName[df]])
           
    for df in range(len(wald)):
        pval,mins=markov(func,wald[df],psi,offDiag,numCores)
        log('markov {} : {} dataset {} min'.format(methName[-1],dataSetName[df],mins))

        plotPower(pval,methName[-1]+'-'+dataSetName[df],cols=[methName[-1]])  
    
        pvalSet[dataSetName[df]]+=[pval.reshape(-1,1)]

    for df in range(len(wald)):
        pdb.set_trace()
        plotPower(np.concatenate(pvalSet[dataSetName[df]],axis=1),dataSetName[df],cols=methName)

    return()

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
    
    numCores=cpu_count()
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
        psiDF=psi(calD,vZ,numLam=numLam,minEta=minEta,numCores=16,eps=eps).compute()
        np.savetxt('psi',psiDF,delimiter='\t')
    if 'loadPsi' in fit:
        psiDF=np.loadtxt('psi',delimiter='\t',dtype=[('lam','float64'),('eta','float64')])

    if 'plot' in fit:
        plots(ds,vZ,psiDF,offDiag,refReps,maxRefReps,numCores,dsNames)

    #stat.plot(gbj('GBJ',wald,numCores=3,offDiag=offDiag),'gbj')
    #plotPower(gbj('GHC',wald,numCores=3,offDiag=offDiag),'ghc')
    

ops={
    'seed':None,
    'numKSnps':1000,
    'calD':0.2,
    'eta':0.3
}

ctrl={
    'numSubjects':500,
    'numH0Snps':1000,
    'numH1Snps':300,
    'numTraits':200,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'rho':1,
    'refReps':int(1e6),
    'maxRefReps':int(1e5),
    'minEta':1e-10,
    'numLam':5e3,
    'mu':10,
    'eps':1e-10,
    'fit':['loadH0','loadH1','plot','loadPsi'],
    'n_assoc':[10,30,50,70,80,100,150]
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

diagnostics(parms['seed'])
log(parms)

'''
increase the granularity of precompute of regions. possibly just need increase in number of ref reps.
'''
''' 
change uniform eta to beta(1,10)
'''
''' 
p.s. figure out change in pvals from change in ref distn (i.e. debug power code) confirm MC code for power calc
'''
myMain(parms)

git('{} mice, {} snps, {} traits, subsample {}, rho {}'.format(parms['numSubjects'],parms['numH0Snps'],
    parms['numTraits'],parms['pedigreeMult'],parms['rho']))
