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

from ail.opPython.DB import *
from ail.genPython.makePSD import *

def makeSimPedFiles(parms):
    print('simSetup')
    
    maxSnpGen=parms['maxSnpGen']
    name=parms['name']
    local=parms['local']
    linBatch=parms['linBatch']
    traitChr=parms['traitChr']
    snpChr=parms['snpChr']
    H1SnpSize=parms['H1SnpSize']
    H2SnpSize=parms['H2SnpSize']
    maxSnpGen=parms['maxSnpGen']
    nameForGRM=parms['nameForGRM']
    etaGRM=parms['etaGRM']
    etaError=parms['etaError']
    numCores=parms['numCores']
    response=parms['response']
    simGRM=parms['simGRM']
    snpFile=parms['snpFile']
    simSnps=parms['simSnps']
                
    DBCreateFolder('ped',parms)
    DBCreateFolder('geneDrop',parms)
    DBCreateFolder('LZCorr',parms)
   
    ################################################### from runAnalysis ###################################################
    
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()
    np.savetxt('ped/mouseIds',mouseIds,delimiter='\t')

    robjects=result = pyreadr.read_r(local+'data/allMouseGenesCoords.RData')
    mouseGenes=robjects['mouseGenes']
    mouseGenes=mouseGenes[mouseGenes['chrom'].isin([str(x) for x in traitChr])]

    robjects=result = pyreadr.read_r(local+'data/qnormPhenos.G50_56.RData')
    allIds=robjects['qnormPhenos']['id'].values.flatten().astype(int).tolist()

    traitData=pd.DataFrame({'trait':mouseGenes['gene_name'],'chr':mouseGenes['chrom'],
        'Mbp':((mouseGenes['cds_start']+mouseGenes['cds_end'])/2).astype(int)})   
    traitData=traitData.loc[traitData['trait'].isin(traits.columns)]
    traitData.to_csv('ped/traitData',index=False,sep='\t')
    
    ################################################### snps ped ###################################################
    
    if not simSnps:
        snps=pd.read_csv(local+'data/'+snpFile,sep='\t',header=None,index_col=None)
        snps.columns=['chr','Mbp','Major','Minor']+allIds
        snpMajor=snps['Major'].values
        snps=snps[mouseIds]
        snps=snps.iloc[np.linspace(0,len(snps)-1,H1SnpSize+H2SnpSize).astype(int),:].T
    else:
        pd.DataFrame({'id':[str(Id)+'.1' for Id in mouseIds]}).to_csv('geneDrop/sampleIds.txt',index=False,header=False)

        pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
            'geneDrop/parms.txt',index=False,header=None)

        '''
        snps=[]
        numSnps=0
        size=H1SnpSize+H2SnpSize
        while numSnps<size:
            newAdd=min(maxSnpGen,size-numSnps)
            pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
                'geneDrop/map.txt',sep='\t',index=False)
            cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
            subprocess.run(cmd)

            val=pd.read_csv('geneDrop/geneDrop.geno_true',header=0,sep='\t').iloc[:,4:]
            pdb.set_trace()
            valMAF=np.concatenate([(col.str.split(' ',expand=True).astype(int)>2).sum(axis=1).values.reshape(-1,1) for ind,
                col in val.iteritems()],axis=1)
            val=pd.concat([col.str.split(' ',expand=True).replace({'1':'A','2':'A','3':'G','4':'G'}).apply(lambda x:' '.join(x),axis=1)
                for ind,col in val.iteritems()],axis=1)
            pdb.set_trace()

            maf=np.mean(valMAF,axis=1)/2
            maf=np.minimum(maf,1-maf)
            val=val.loc[maf>.1,:]

            numSnps+=val.shape[0]
            print(numSnps,val.shape,flush=True)
            print('removed '+str(newAdd-val.shape[0])+' snps',flush=True)

            snps+=[val]

        snps=pd.concat(snps,axis=0).T
        '''
        
    snpData=pd.DataFrame({'chr':[1]*H1SnpSize+[2]*H2SnpSize,'ID':range(H1SnpSize+H2SnpSize),
        'genetic dist': 0,'Mbp':range(H1SnpSize+H2SnpSize)})
    snpData.to_csv('ped/snpData',index=False,sep='\t')

    pdb.set_trace()
    bimSnps=pd.DataFrame({'Family ID':0,'Individual ID':range(len(snps)),'Paternal ID':0,'Maternal ID':0,'Sex':0,'Phenotype':0})

    for snp in snpChr:
        pd.concat([bimSnps,snps.loc[:,snpData['chr'].values==snp]],axis=1).to_csv('ped/ail-'+str(snp)+'.ped',header=False,
            index=False,sep='\t')   
        snpData[snpData['chr'].values==snp].to_csv('ped/ail-'+str(snp)+'.map',header=False,index=False,sep='\t')  

    pd.concat([bimSnps,snps],axis=1).to_csv('ped/ail.ped',header=False,index=False,sep='\t')   
    snpData.to_csv('ped/ail.map',header=False,index=False,sep='\t')  
    
    ################################################### grm ped ###################################################

    if not simGRM:
        os.symlink('../../'+nameForGRM+'/ped/eigen-all','ped/eigen-1')
        os.symlink('../../'+nameForGRM+'/ped/eigen-all','ped/eigen-2')
        os.symlink('../../'+nameForGRM+'/ped/eigen-all','ped/eigen-all')
        os.symlink('../../'+nameForGRM+'/LZCorr/Lgrm-all','LZCorr/Lgrm-all')
    else:        
        pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits)),'dummy':range(len(traits))}).to_csv(
            'ped/ail.phe',header=False,index=False,sep='\t')

        cmd=[local+'ext/fastlmmc','-file','ped/ail','-fileSim','ped/ail','-pheno','ped/ail.phe','-runGwasType','NORUN',
             '-maxThreads',str(numCores),'-simOut','ped/simGrm','-eigenOut','ped/eigen-all']
        subprocess.call(cmd)
        
        shutil.copytree('ped/eigen-all', 'ped/eigen-1')
        shutil.copytree('ped/eigen-all', 'ped/eigen-2')

        grm=pd.read_csv('ped/simGrm',sep='\t',index_col=0,header=0).values
        np.savetxt('LZCorr/Lgrm-all',makePSD(grm),delimiter='\t')

    os.symlink('../../'+nameForGRM+'/LZCorr/Lgrm-all','LZCorr/Lgrm-'+nameForGRM)
    os.symlink('../../'+nameForGRM+'/LZCorr/LTraitCorr','LZCorr/LRawTraitCorr')
    LgrmAll=np.loadtxt('LZCorr/Lgrm-all',delimiter='\t')
    
    ################################################### gen Y ###################################################

    LRawTraitCorr=np.loadtxt('LZCorr/LRawTraitCorr',delimiter='\t')
    traitSize=[len(LgrmAll),len(LRawTraitCorr)]
    
    Y=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LRawTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LRawTraitCorr.T)
    
    pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(traits))}),pd.DataFrame(Y)],axis=1).to_csv(
        'ped/ail.phe',header=False,index=False,sep='\t')

    YCorr=np.corrcoef(Y,rowvar=False)
    np.savetxt('LZCorr/LTraitCorr',makePSD(YCorr),delimiter='\t')

    print('genY',flush=True)
    
    ################################################### cov ped ###################################################
        
    pd.DataFrame({'Family ID':0,'Individual ID':range(len(Y)),'Intercept':1}).to_csv('ped/cov.phe',header=False,index=False,sep='\t')
    
    return()
