import pandas as pd
import numpy as np
import pdb
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pyreadr
import subprocess
import shutil
import random
import matplotlib.pyplot as plt

from ail.opPython.DB import *
from ail.genPython.makePSD import *

def plotGrm(parms):
    print('simSetup')
    
    name=parms['name']
    local=parms['local']
    numCores=parms['numCores']
    response=parms['response']
    snpFile=parms['snpFile']
   
    ################################################### from runAnalysis ###################################################
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()
    np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()

    print('loading real snp data',flush=True)
    snps=pd.read_csv(local+'data/'+snpFile,sep='\t',header=None,index_col=None)
    snps.columns=['chr','Mbp','Major','Minor']+allIds
    snpMajor=snps['Major'].values.flatten().tolist()
    snps=snps[mouseIds].T.reset_index(drop=True)

    bimSnps=pd.DataFrame({'Family ID':0,'Individual ID':range(len(snps)),'Paternal ID':0,'Maternal ID':0,'Sex':0,'Phenotype':0})
    
    pd.concat([bimSnps,snps],axis=1).to_csv('ped/ail-real.ped',header=False,index=False,sep='\t')   
    pd.DataFrame({'chr':[1]*snps.shape[1],'ID':range(snps.shape[1]),'genetic dist': 0,'Mbp':0}).to_csv(
        'ped/ail-real.map',header=False,index=False,sep='\t')

    cmd=[local+'ext/fastlmmc','-file','ped/ail-real','-fileSim','ped/ail-real','-pheno','ped/ail.phe','-runGwasType','NORUN',
         '-maxThreads',str(numCores),'-simOut','ped/grmReal']
    subprocess.call(cmd)

    grmReal=pd.read_csv('ped/grmReal',sep='\t',index_col=0,header=0).values
    grmSim=pd.read_csv('ped/simGrm',sep='\t',index_col=0,header=0).values
    
    diagReal=np.diag(grmReal)
    diagSim=np.diag(grmSim)
    
    offDiagReal=grmReal[np.triu_indices(len(grmReal),1)]
    offDiagSim=grmSim[np.triu_indices(len(grmSim),1)]
    
    fig,axs=plt.subplots(2,1)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(40,forward=True)
    
    axs[0].scatter(diagReal,diagSim)
    mMax=max(max(diagReal),max(diagSim))
    mMin=min(min(diagReal),min(diagSim))
    axs[0].set_xlim([mMin,mMax])
    axs[0].set_ylim([mMin,mMax])
    axs[0].set_xlabel('real')
    axs[0].set_ylabel('sim')
    axs[0].plot([mMin,mMax],[mMin,mMax],ls='--',c='.3')
    axs[0].set_title('diag')
    
    axs[1].scatter(offDiagReal,offDiagSim)
    mMax=max(max(offDiagReal),max(offDiagSim))
    mMin=min(min(offDiagReal),min(offDiagSim))
    axs[1].set_xlim([mMin,mMax])
    axs[1].set_ylim([mMin,mMax])
    axs[1].set_xlabel('real')
    axs[1].set_ylabel('sim')
    axs[1].plot([mMin,mMax],[mMin,mMax],ls='--',c='.3')
    axs[1].set_title('off-diag')
    
    fig.savefig('diagnostics/grm.png')
    pdb.set_trace()
    return()