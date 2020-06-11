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

def myMain(parms,vZ=None):
    numDataSnps=parms['numDataSnps']
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
    betaParm=parms['betaParm']
    
    eps=parms['eps']
    maxIter=parms['maxIter']
            
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitY' in fit:
        miceRange=np.random.choice(208,int(pedigreeMult*208),replace=False)    

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
        np.savetxt('miceRange',miceRange,delimiter='\t')
    else:
        Y=np.loadtxt('Y',delimiter='\t')
        K=np.loadtxt('K',delimiter='\t')        
        M=np.loadtxt('M',delimiter='\t');M=M.reshape(len(M),-1)  
        miceRange=np.loadtxt('miceRange',delimiter='\t').astype(int)

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitWaldH0' in fit:
        snps=makePedigreeSnps(numSubjects,miceRange,numDataSnps,numCores)
        QS=economic_qs(K)
        wald,eta=runLimix(Y,QS,np.ones([numSubjects,1]),snps,0.9999)

        np.savetxt('wald',wald,delimiter='\t')
        np.savetxt('eta',eta,delimiter='\t')
        np.savetxt('snps',snps,delimiter='\t')
    else:
        wald=np.loadtxt('wald',delimiter='\t')
        eta=np.loadtxt('eta',delimiter='\t')
        snps=np.loadtxt('snps',delimiter='\t')        
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitVz' in fit:
        vZ=np.corrcoef(wald,rowvar=False)
        offDiag=vZ[np.triu_indices(vZ.shape[1],1)]   
        np.savetxt('vZ',vZ,delimiter='\t')
    else:
        vZ=np.loadtxt('vZ',delimiter='\t')        
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitWaldH1' in fit:
        wald=runH1(betaParm,n_assoc,wald,Y,K,M,snps,eta)
        np.savetxt('waldH'+str(n_assoc),wald,delimiter='\t')
    elif 'loadWaldH1' in fit:
        wald=np.loadtxt('waldH'+str(n_assoc),delimiter='\t')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    if 'fitPsi' in fit:
        psiDF=psi(calD,vZ,numLam,minEta,numCores,eps,maxIter,numHermites).compute()
        np.savetxt('psi',psiDF,delimiter='\t')
    else:
        psiDF=np.loadtxt('psi',delimiter='\t',dtype=[('lam','float64'),('eta','float64')])
    
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
        'markov':partial(markov,psiDF)
    }

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    if 'fitRef' in fit:
        ref={key:genRef(f,refReps,maxRefReps,vZ,key) for key,f in mcFuncs.items() if key in mcMethodNames}
        for key in mcMethodNames:
            np.savetxt('ref-'+key,ref[key],delimiter='\t')
    else:
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

    if 'plot' in fit:
        wald=-np.sort(-np.abs(wald))
        pvals=[]
        for method in mcMethodNames+markovMethodNames:
            pval=pvalFuncs[method](wald).reshape(-1,1)       
            plot(pval,method,cols=[method])        
            pvals+=[pval]
           
        power=plot(np.concatenate(pvals,axis=1),'H'+str(n_assoc),cols=mcMethodNames+markovMethodNames)
    else:
        power=None
        
    return(power)
    
ops={
    'seed':None,
    'numKSnps':1000,
    'calD':0.2,
    'eta':0.3
}

'''
redo with one Y_0 for all n_assoc (all n_assoc have same Y_0 just regenerated from previous simulations)
keep parms the same beta across replications of entire simulation
great to do it 10 times if possible
leave beta the same rather than searching for new ones
only really need the capital H plots
type 1 all on one plot
flesh out section on what is going in
put in type 1 plot and table
pg 18,19 update
pg 20 put in full language
replace multiple type 1 plots with 1 all included plot
redo y, grm snps, v(z), keep the same ref across n_assoc (but using new V(Z))
# find average LValue per D
# find area under C
'''
ctrl={
    'numSubjects':1200,
    'numDataSnps':1000,
    'numTraits':1200,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'rho':1,
    'refReps':int(1e6),
    'maxRefReps':int(1e5),
    'minEta':1e-12,
    'numLam':1e3,
    'eps':1e-13,
    'maxIter':1e2,
    'numHermites':150,
    'numCores':cpu_count(),
    'mcMethodNames':['ELL','cpma','score','storey','minP'],
    'markovMethodNames':[]
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()

createDiagnostics(parms['seed'])
log(parms)

betaParms=np.array([(500,1.3)],dtype=[('n_assoc','int'),('beta','float64')])
#np.array([[1,3.194],[2,3.125],[4,2.89],[10,2.568],[50,2],[150,1.53],[500,1.38],[800,1.15]])
#_=myMain({**parms,'n_assoc':None,'betaParm':None,'fit':['fitWaldH0','fitY','fitVz']}) # create waldH0 for est VZ
#_=myMain({**parms,'n_assoc':None,'numDataSnps':300,'betaParm':None,'fit':['fitWaldH0','fitPsi','fitRef']}) # create waldH1
for n_assoc,beta in betaParms:
    power=myMain({**parms,'betaParm':beta,'n_assoc':n_assoc,'fit':['plot','fitWaldH1']},None)

    git('n_assoc {}, beta {}, maxPower {}'.format(n_assoc,beta,np.max(power)))
