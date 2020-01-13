import matplotlib
matplotlib.use('agg')
import warnings
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from opPython.setupFolders import *

from dataPrepPython.makePedFiles import *
from dataPrepPython.genZScores import *

from datetime import datetime
from scipy.stats import t, norm

local=os.getcwd()+'/'

snp=1
trait=1
predsList=['sex','nonLinBatch']
chrLoc=list(range(100))

parms={
    'local':local,
    'name':'fastVsGemma-'+str(snp)+'-'+str(trait)+'-'+str(predsList)+'/',
    'numCores':cpu_count(),
    'traitChr':[trait],
    'snpChr':[snp],
    'predsList':predsList,
    'remList':[],
    'logName':'log/fastVsGemma-'+str(datetime.now()),
    'simLearnType':'Full',
    'response':'hipExp',
    'quantNormalizeExpr':False,
    'verbose':False,
    'chrLoc':chrLoc
}

####################################################################################################

setupFolders(parms,'fastVsGemma')

####################################################################################################

print('make peds',flush=True)
makePedFiles({**parms,'fastGrm':False})

####################################################################################################

os.symlink('eigen-all','grm/eigen-'+str(snp))
os.symlink('gemma-all','grm/gemma-'+str(snp))

####################################################################################################

print('Create ZScores',flush=True)

DBCreateFolder('score',parms)
DBCreateFolder('holds',parms)
genZScores({**parms,'fastlmm':False})
DBCreateFolder('holds',parms)
genZScores({**parms,'fastlmm':True})

####################################################################################################

print('load scores',flush=True)
traitData=pd.read_csv('ped/traitData',sep='\t',index_col=None,header=0)
snpData=pd.read_csv('ped/snpData',sep='\t',index_col=None,header=0)

nameParm=str(snp)+'-'+str(trait)

gemma=np.loadtxt('score/gemma-AltLogLike-'+nameParm,delimiter='\t').flatten()
fast=np.loadtxt('score/fast-AltLogLike-'+nameParm,delimiter='\t').flatten()

df=pd.DataFrame({
    'trait':np.concatenate([np.arange(sum(traitData['chr']==trait)).reshape(1,-1)]*sum(snpData['chr']==snp),axis=0).flatten(),
    'snp':np.concatenate([np.arange(sum(snpData['chr']==snp)).reshape(-1,1)]*sum(traitData['chr']==trait),axis=1).flatten(),
    'gemma-AltLogLike':np.loadtxt('score/gemma-AltLogLike-'+nameParm,delimiter='\t').flatten(),
    'fast-AltLogLike':np.loadtxt('score/fast-AltLogLike-'+nameParm,delimiter='\t').flatten(),
    'gemma-beta':np.loadtxt('score/gemma-beta-'+nameParm,delimiter='\t').flatten(),
    'fast-beta':np.loadtxt('score/fast-beta-'+nameParm,delimiter='\t').flatten(),
    'gemma-se':np.loadtxt('score/gemma-se-'+nameParm,delimiter='\t').flatten(),
    'fast-se':np.loadtxt('score/fast-se-'+nameParm,delimiter='\t').flatten(),
    'gemma-pLRT':np.loadtxt('score/gemma-pLRT-'+nameParm,delimiter='\t').flatten(),
    'gemma-pWald':np.loadtxt('score/gemma-pWald-'+nameParm,delimiter='\t').flatten(),
    'gemma-pZ':2*norm.sf(np.abs(np.loadtxt('score/gemma-waldStat-'+nameParm,delimiter='\t').flatten())),
    'gemma-pT':2*t.sf(np.abs(np.loadtxt('score/gemma-waldStat-'+nameParm,delimiter='\t').flatten()),207),
    'fast-pLRT':np.loadtxt('score/fast-pLRT-'+nameParm,delimiter='\t').flatten(),
    'fast-pWald':np.loadtxt('score/fast-pWald-'+nameParm,delimiter='\t').flatten(),
    'fast-pZ':2*norm.sf(np.abs(np.loadtxt('score/fast-waldStat-'+nameParm,delimiter='\t').flatten())),
    'fast-pT':2*t.sf(np.abs(np.loadtxt('score/fast-waldStat-'+nameParm,delimiter='\t').flatten()),207)
})   

names=df.columns[8:].tolist()

####################################################################################################

fig, axs = plt.subplots(2,1,dpi=50,tight_layout=True)   
fig.set_figwidth(10*1,forward=True)
fig.set_figheight(10*2,forward=True)

df[['snp']+names].groupby('snp').min().plot(legend=True,logy=True,ax=axs[0])
df[['trait']+names].groupby('trait').min().plot(legend=True,logy=True,ax=axs[1])
fig.savefig('diagnostics/mins.png',bbox_inches='tight')
plt.close('all')

####################################################################################################

B=len(names)
fig, axs = plt.subplots(int(B*(B-1)/2),2,dpi=50,tight_layout=True)   
fig.set_figwidth(10*2,forward=True)
fig.set_figheight(10*int(B*(B-1)/2),forward=True)

print('plot',flush=True)
count=0
t_df=df[names]
for i in range(B-1):
    for j in range(i+1,B):
        mMax=max(max(t_df.iloc[:,i]),max(t_df.iloc[:,j]))
        mMin=min(min(t_df.iloc[:,i]),min(t_df.iloc[:,j]))

        axs[count,0].scatter(t_df.iloc[:,i],t_df.iloc[:,j])
        axs[count,0].set_ylabel(names[j])
        axs[count,0].set_xlabel(names[i])
        axs[count,0].set_xlim([mMin,mMax])
        axs[count,0].set_ylim([mMin,mMax])
        axs[count,0].plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")
        axs[count,0].set_title(names[j]+' ~ '+names[i])
        
        tMax=mMax
        mMax=-np.log10(mMin)
        mMin=-np.log10(tMax)
        
        axs[count,1].scatter(-np.log10(t_df.iloc[:,i]),-np.log10(t_df.iloc[:,j]))
        axs[count,1].set_ylabel(names[j])
        axs[count,1].set_xlabel(names[i])
        axs[count,1].set_xlim([mMin,mMax])
        axs[count,1].set_ylim([mMin,mMax])
        axs[count,1].plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")
        axs[count,1].set_title('log '+names[j]+' ~ log '+names[i])
        
        count+=1

print('write plot',flush=True)
fig.savefig('diagnostics/fastVsGemma.png',bbox_inches='tight')
plt.close('all')

####################################################################################################

fig, axs = plt.subplots(2,1,dpi=50,tight_layout=True)   
fig.set_figwidth(10*1,forward=True)
fig.set_figheight(10*2,forward=True)

x=df['gemma-AltLogLike']
y=df['fast-AltLogLike']
mMax=max(x.max(),y.max())
mMin=min(x.min(),y.min())
axs[0].scatter(x,y)
axs[0].set_xlabel('gemma-AltLogLike')
axs[0].set_ylabel('fast-AltLogLike')
axs[0].set_xlim([mMin,mMax])
axs[0].set_ylim([mMin,mMax])
axs[0].plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")

x=df['gemma-AltLogLike']-df['fast-AltLogLike']
y=df['gemma-pWald']-df['fast-pWald']
axs[1].scatter(x,y)
axs[1].set_xlabel('AltLogLike diff')
axs[1].set_ylabel('pWald diff')

fig.savefig('diagnostics/altlike.png',bbox_inches='tight')
plt.close('all')

####################################################################################################

fig, axs = plt.subplots(2,1,dpi=50,tight_layout=True)   
fig.set_figwidth(10*1,forward=True)
fig.set_figheight(10*2,forward=True)

x=df['gemma-beta']
y=df['fast-beta']
mMax=max(x.max(),y.max())
mMin=min(x.min(),y.min())
axs[0].scatter(x,y)
axs[0].set_xlabel('gemma-beta')
axs[0].set_ylabel('fast-beta')
axs[0].set_xlim([mMin,mMax])
axs[0].set_ylim([mMin,mMax])
axs[0].plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")

x=df['gemma-beta']-df['fast-beta']
y=df['gemma-pWald']-df['fast-pWald']
axs[1].scatter(x,y)
axs[1].set_xlabel('beta diff')
axs[1].set_ylabel('pWald diff')

fig.savefig('diagnostics/beta.png',bbox_inches='tight')
plt.close('all')

####################################################################################################

fig, axs = plt.subplots(2,1,dpi=50,tight_layout=True)   
fig.set_figwidth(10*1,forward=True)
fig.set_figheight(10*2,forward=True)

x=df['gemma-se']
y=df['fast-se']
mMax=max(x.max(),y.max())
mMin=min(x.min(),y.min())
axs[0].scatter(x,y)
axs[0].set_xlabel('gemma-se')
axs[0].set_ylabel('fast-se')
axs[0].set_xlim([mMin,mMax])
axs[0].set_ylim([mMin,mMax])
axs[0].plot([mMin,mMax],[mMin,mMax],ls="--", c=".3")

x=df['gemma-se']-df['fast-se']
y=df['gemma-pWald']-df['fast-pWald']
axs[1].scatter(x,y)
axs[1].set_xlabel('se diff')
axs[1].set_ylabel('pWald diff')

fig.savefig('diagnostics/se.png',bbox_inches='tight')
plt.close('all')
