import pandas as pd
import numpy as np
import subprocess
import pdb
import os
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from scipy.stats import chi2

def snpR2HistsHelp(snp,parms): 
    scratchDir=parms['scratchDir']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    
    if os.path.isfile(scratchDir+'z-squared-R2-'+snp+'-trans.csv'):
        return()

    #np.savetxt(scratchDir+'z-squared-R2-'+snp+'-cis.csv',np.array([]),delimiter=',')
    #np.savetxt(scratchDir+'z-squared-R2-'+snp+'-trans.csv',np.array([]),delimiter=',')
    
    trans=[] 
    for trait in traitChr:
        df=np.loadtxt(scratchDir+'z-'+snp+'-'+trait+'.csv',delimiter=',').astype('float16')**2

        if snp==trait:
            cis=df
        else:
            trans+=[df]
            
    trans=np.concatenate(trans,axis=1)

    print('writing R2',snp,flush=True)
    np.savetxt(scratchDir+'z-squared-R2-'+snp+'-cis.csv',np.median(cis,axis=1),delimiter=',')
    np.savetxt(scratchDir+'z-squared-R2-'+snp+'-trans.csv',np.median(trans,axis=1),delimiter=',')

def zSquaredHistsHelp(trait,parms): 
    scratchDir=parms['scratchDir']
    snpChr=parms['snpChr']
    
    if os.path.isfile(scratchDir+'z-squared-'+trait+'-trans.csv'):
        return()

    trans=[] 
    cis=pd.DataFrame()
    for snp in snpChr:
        #print('loading snps '+snp+' for trait '+trait)
        df=np.loadtxt(scratchDir+'z-'+snp+'-'+trait+'.csv',delimiter=',').astype('float16')**2

        if snp==trait:
            cis=df
        else:
            trans+=[df]
            
    trans=np.concatenate(trans,axis=0)

    print('writing zsquared',trait,flush=True)
    if trait in snpChr:
        np.savetxt(scratchDir+'z-squared-'+trait+'-cis.csv',np.mean(cis,axis=0),delimiter=',')
    np.savetxt(scratchDir+'z-squared-'+trait+'-trans.csv',np.mean(trans,axis=0),delimiter=',')

def zSquaredHists(parms):
    plotsDir=parms['plotsDir']
    scratchDir=parms['scratchDir']
    
    snpChr=parms['snpChr']
    traitChr=parms['traitChr']
        
    snpData=pd.read_csv(scratchDir+'snpData.csv')
    traitData=pd.read_csv(scratchDir+'traitData.csv')
    
    futures=[]
    #snpR2HistsHelp('chr1',parms)
    with ProcessPoolExecutor(parms['smallCpu']) as executor: 
        for trait in traitChr:
            futures.append(executor.submit(zSquaredHistsHelp,trait,parms))
            
        if parms['numPCs']>0:
            for snp in snpChr:
                futures.append(executor.submit(snpR2HistsHelp,snp,parms))

        wait(futures,return_when=ALL_COMPLETED)    
    
    # hist of medians
    fig,axs=plt.subplots(len(traitChr),1,dpi=50)
    fig.set_figwidth(20,forward=True)
    fig.set_figheight(190,forward=True)
    fig.tight_layout()

    for i in range(len(traitChr)):
        trait=traitChr[i]
        
        trans=np.loadtxt(scratchDir+'z-squared-'+trait+'-trans.csv',delimiter=',')

        axs[i].hist(trans,label='trans',density=True,bins=100,alpha=.5,log=True)
        axs[i].axvline(x=np.mean(trans),label='trans',linewidth=4)

        if trait in snpChr:
            cis=np.loadtxt(scratchDir+'z-squared-'+trait+'-cis.csv',delimiter=',')
            axs[i].hist(cis,label='cis',density=True,bins=20,alpha=.5,log=True)
            axs[i].axvline(x=np.mean(cis),label='cis',linewidth=4)

        axs[i].legend()
        
    fig.savefig(plotsDir+'z_squared.png',bbox_inches='tight')
    
    # plot of median ~ R^2
    if parms['numPCs']>0:
        q=chi2.ppf(.50,1)
        
        fig,axs=plt.subplots(len(snpChr),4,dpi=50)
        fig.set_figwidth(60,forward=True)
        fig.set_figheight(190,forward=True)

        R2=np.loadtxt(scratchDir+'snpR2.csv',delimiter=',')

        for i in range(len(snpChr)):
            snp=snpChr[i]
            deltaTrans=1.96*np.sqrt(.5*.5/(np.sum(traitData['chr']!=snp)*chi2.pdf(q,1)**2))
            deltaCis=1.96*np.sqrt(.5*.5/(np.sum(traitData['chr']==snp)*chi2.pdf(q,1)**2))
            R2Loc=np.sqrt(R2[snpData['chr']==snp])

            j=0
            trans=np.loadtxt(scratchDir+'z-squared-R2-'+snp+'-trans.csv',delimiter=',')
            axs[i,j].hist(trans,label='trans',density=True,bins=100,alpha=.5,log=True)
            axs[i,j].axvline(x=np.mean(trans),label='obs median',linewidth=4)
            axs[i,j].axvline(x=q+deltaTrans,label='band',color='k')
            axs[i,j].axvline(x=q-deltaTrans,color='k')
            axs[i,j].legend()

            j+=1
            cis=np.loadtxt(scratchDir+'z-squared-R2-'+snp+'-cis.csv',delimiter=',')
            axs[i,j].hist(cis,label='cis',density=True,bins=20,alpha=.5,log=True)
            axs[i,j].axvline(x=np.mean(cis),label='obs median',linewidth=4)
            axs[i,j].axvline(x=q+deltaCis,label='band',color='k')
            axs[i,j].axvline(x=q-deltaCis,color='k')
            axs[i,j].legend()

            j+=1
            axs[i,j].scatter(R2Loc,trans,label='trans')
            axs[i,j].axhline(y=q+deltaTrans,label='band',color='k')
            axs[i,j].axhline(y=q-deltaTrans,color='k')
            axs[i,j].set_ylabel('Median Z**2',fontsize=10)
            axs[i,j].set_xlabel('sqrt(R^2)')
            axs[i,j].set_title('trans- '+str(int(100*np.mean((trans<q+deltaTrans)&(trans>q-deltaTrans))))+'% in band')
            axs[i,j].set_aspect('equal')
            axs[i,j].legend()

            j+=1
            axs[i,j].scatter(R2Loc,cis,label='cis')
            axs[i,j].axhline(y=q+deltaCis,label='band',color='k')
            axs[i,j].axhline(y=q-deltaCis,color='k')
            axs[i,j].set_ylabel('Median Z**2',fontsize=10)
            axs[i,j].set_xlabel('sqrt(R^2)')
            axs[i,j].set_title('cis- '+str(int(100*np.mean((cis<q+deltaCis)&(cis>q-deltaCis))))+'% in band')
            axs[i,j].set_aspect('equal')
            axs[i,j].legend()

        fig.savefig(plotsDir+'z_squared_R2.png',bbox_inches='tight')

        plt.close('all')
