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
from plotPython.tablePower import *
from utility import *
from plotPython.myHist import *
from limix.model.lmm import LMM
from limix.stats import linear_kinship
from dataPrepPython.makePedigreeSnps import *
from numpy_sugar.linalg import economic_qs
from statsPython.szs import *
from statsPython.fdr import *
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
    effectSize=parms['effectSize']
    
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

    if 'runLimix' in fit:
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
    
    if 'computeH1' in fit:
        wald=runH1(effectSize,n_assoc,wald,Y,K,M,snps,eta,numCores)
        np.savetxt('waldH'+str(n_assoc),wald,delimiter='\t')
    elif 'loadComputeH1' in fit:
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
        'CPMA':partial(cpma,numCores),
        'sumZ^2':partial(szs,numCores),
        'FDR':partial(fdr,numCores),
        'minP':partial(minP,numCores)
    }
    
    markovFuncs={
        'GBJ':partial(gbjLoop,'GBJ'),
        'GHC':partial(gbjLoop,'GHC'),
        'ELL-analytic':partial(markov,psiDF)
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

    if 'fitStats' in fit:
        wald=-np.sort(-np.abs(wald))
        pvals=[]
        for method in mcMethodNames+markovMethodNames:
            pval=pvalFuncs[method](wald).reshape(-1,1)       
            pvals+=[pval]
           
        pvals=np.concatenate(pvals,axis=1)
 
    if 'plotPower' in fit:
        plotPower(pvals,'H'+str(n_assoc),cols=mcMethodNames+markovMethodNames)

    if 'tablePower' in fit:
        power=tablePower(pvals,'H'+str(n_assoc),cols=mcMethodNames+markovMethodNames)
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
xx=pd.read_csv('power.tsv',sep='&')
xx['n_assoc']=xx['n_assoc']/1200
xx=xx.set_index('n_assoc')
xx=(xx.T/xx.T.max(axis=0)).T
fig, axs = plt.subplots(1,1,dpi=50)
plt.rc('font', size=16)       
plt.rc('axes', labelsize=20)   
fig.set_figheight(10,forward=True)
fig.set_figwidth(10,forward=True)
xx.plot(ax=axs,ylim=[0,1.1],xlim=[-.1,.5],marker='o', linestyle='dotted',legend=True,logx=False)
axs.set_xlabel('% of traits associated')
axs.set_ylabel('Power (as % of most powerful method)')
fig.savefig('power.png',bbox_inches='tight')
'''
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
    'numTraits':10000,
    'pedigreeMult':.1,
    'snpParm':'geneDrop',
    'rho':1,
    'refReps':int(1e6),
    'maxRefReps':int(1e5),
    'minEta':1e-12,
    'numLam':1e2,
    'eps':1e-13,
    'maxIter':1e2,
    'numHermites':150,
    'numCores':cpu_count(),
    'mcMethodNames':['ELL','CPMA','sumZ^2','FDR','minP'],
    'markovMethodNames':[]#'ELL-analytic']
}

#######################################################################################################

parms={**ctrl,**ops}
setupFolders()
coeff=[0,3.7]
nList=[4]

h0Snps=10000
h1Snps=1000

createDiagnostics(parms['seed'])

#_=myMain({**parms,'n_assoc':None,'effectSize':None,'numDataSnps':h0Snps,'fit':['runLimix','fitY','fitVz','fitPsi','fitRef']}) 
#_=myMain({**parms,'n_assoc':None,'effectSize':None,'numDataSnps':h1Snps,'fit':['runLimix']}) 

coeff=[1.7,2.018711487875]
n_assoc=500
power=[10]
while np.abs(np.max(power)-.73)>0.1:
    newBeta=np.mean(coeff)
    power=myMain({**parms,'effectSize':newBeta,'n_assoc':n_assoc,'numDataSnps':None,'fit':['fitStats','computeH1','tablePower']})

    maxPower=max(power)
    if maxPower>0.73:
        coeff[1]=newBeta
    else:
        coeff[0]=newBeta
    
    log('n_assoc {}, newBeta {}, power {}'.format(n_assoc,newBeta,maxPower))

git('power {}'.format(n_assoc))

'''
createDiagnostics(parms['seed'])
log(parms)

nList=[1,2,4,10,50,150,500]
lowBeta=[3.433125,3.356933594,3.2627020872031256,3.076430504,2.634421806,2.299383806,]

power=[]
for run in range(10):
    _=myMain({**parms,'n_assoc':None,'effectSize':None,'numDataSnps':h0Snps,'fit':['runLimix','fitY','fitVz','fitPsi','fitRef']}) 
    _=myMain({**parms,'n_assoc':None,'effectSize':None,'numDataSnps':h1Snps,'fit':['runLimix']}) 
    
    for n in range(len(nList)):   
        n_assoc=nList[n]
        val=np.append(np.array([run,n_assoc,lowBeta[n]]),myMain({**parms,'effectSize':lowBeta[n],'n_assoc':n_assoc,
            'numDataSnps':None,'fit':['fitStats','computeH1','tablePower']})).reshape(1,-1)
        log('n_assoc {}, run {} : '.format(n_assoc,run)+val.tostring())
        
        print('n_assoc {}, run {} : '.format(n_assoc,run)+val.tostring(),flush=True)

git('power')
'''