import pandas as pd
import numpy as np
import pdb
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pyreadr
import subprocess
import shutil

from ail.opPython.DB import *
from ail.genPython.makePSD import *
from ail.dataPrepPython.genGRM import *

def makeSimPedFiles(parms):
    print('simSetup')
    
    response=parms['response']
    name=parms['name']
    local=parms['local']
    linBatch=parms['linBatch']
    traitChr=parms['traitChr']
    grmSnpChr=parms['grmSnpChr']
    quantNormalizeExpr=parms['quantNormalizeExpr']
    H1SnpSize=parms['H1SnpSize']
    H2SnpSize=parms['H2SnpSize']
    maxSnpGen=parms['maxSnpGen']
    nameForGRM=parms[nameForGRM]
    etaGRM=parms['etaGRM']
    etaError=parms['etaError']
                
    traits=pd.read_csv(local+'data/'+response+'.txt',sep='\t',index_col=0,header=0)
    mouseIds=traits.index.values.flatten().astype(int).tolist()

    ################################################### trait pheno file ###################################################
    
    os.symlink('../'+nameForGRM+'/ped/traitData','ped/traitData')
    os.symlink('../'+nameForGRM+'/LZCorr/Lgrm-all','LZCorr/Lgrm-all')
    os.symlink('../'+nameForGRM+'/LZCorr/LTraitCorr','LZCorr/LRawTraitCorr')
    
    LgrmAll=DBRead('LZCorr/Lgrm-all',parms)
    LRawTraitCorr=DBRead('LZCorr/LRawTraitCorr',parms)
    traitSize=[len(LgrmAll),len(LRawTraitCorr)]
    
    Y=etaGRM*np.matmul(np.matmul(LgrmAll,norm.rvs(size=traitSize)),LRawTraitCorr.T)+etaError*np.matmul(
        norm.rvs(size=traitSize),LRawTraitCorr.T)
    
    DBWrite(Y,name+'ped/Y',parms)

    YCorr=np.corrcoef(Y,rowvar=False)
    DBWrite(makePSD(YCorr),'LZCorr/LTraitCorr',parms)

    pd.concat([pd.DataFrame({'Family ID':0,'Individual ID':range(len(Y))}),traits],axis=1).to_csv(
        'ped/ail.phe',header=False,index=False)

    print('genY',flush=True)
    
    ################################################### snps ped ###################################################
    
    pd.DataFrame({'id':[str(Id)+'.1' for Id in mouseIds]}).to_csv('geneDrop/sampleIds.txt',index=False,header=False)
        
    pd.DataFrame({'parms':[local+'ext/ail_revised.ped.txt','geneDrop/sampleIds.txt','geneDrop/map.txt',0,0]}).to_csv(
        'geneDrop/parms.txt',index=False,header=None)
    
    snpData=pd.DataFrame({'chr':['chr1']*H0SnpSize+['chr2']*H1SnpSize,'Mbp':range(H1SnpSize+H2SnpSize)})
    DBWrite(snpData,'ped/snpData',parms)
    
    pd.DataFrame({'# name':np.arange(1,newAdd+1),'length(cM)':1,'spacing(cM)':2,'MAF':.5}).to_csv(
        'geneDrop/map.txt',sep='\t',index=False)

    grmSnps=pd.DataFrame()
    for snp in range(1,3):
        snps=[]
        numSnps=0
        while numSnps<size:
            newAdd=min(maxSnpGen,size-numSnps)
            cmd=[local+'ext/gdrop','-p','geneDrop/parms.txt','-s',str(random.randint(1,1e6)),'-o','geneDrop/geneDrop']
            subprocess.run(cmd)

            val=pd.read_csv('geneDrop/geneDrop.geno_true',header=0,sep='\t').iloc[:,4:]
            val=np.concatenate([(col.str.split(' ',expand=True).astype(int)>2).sum(axis=1).values.reshape(-1,1) for ind,
                col in val.iteritems()],axis=1)

            maf=np.mean(val,axis=1)/2
            maf=np.minimum(maf,1-maf)
            val=val[maf>.1,:]

            numSnps+=val.shape[0]
            print('removed '+str(newAdd-val.shape[0])+' snps',flush=True)

            snps+=[val]

        bimSnps=pd.DataFrame(np.concatenate(snps,axis=0))
        bimSnps.insert(0,'Phenotype',0)
        bimSnps.insert(0,'Sex',0)
        bimSnps.insert(0,'Maternal ID',0)
        bimSnps.insert(0,'Paternal ID',0)
        bimSnps.insert(0,'Individual ID',range(len(snps)))
        bimSnps.insert(0,'Family ID',0)
        bimSnps.to_csv('ped/ail-'+snp+'.ped',header=False,index=False,sep='\t')   
        
        snpData[snpData['chr']==snp].to_csv('ped/ail-'+snp+'.map',header=False,index=False,sep='\t')  

        cmd=[local+'ext/fastlmmc','-file','ped/ail-'+snp,'-fileSim','ped/ail','-runGwasType','NORUN',
             '-eigenOut','ped/eigen-'+snp,'-extractSim','ped/extractSim-'+snp,'-maxThreads',str(numCores),'-simOut','ped/grm-'+snp]
        subprocess.call(cmd)

        grmSnps=grmSnps.append(bimSnps)
     
    grmSnps.to_csv('ped/ail-all.map',header=False,index=False,sep='\t')
    cmd=[local+'ext/fastlmmc','-file','ped/ail-all','-fileSim','ped/ail-all','-runGwasType','NORUN',
         '-eigenOut','ped/eigen-all','-maxThreads',str(numCores),'-simOut','ped/grm-all']
    subprocess.call(cmd)

    grm=pd.read_csv('ped/grm-'+snp,sep='\t',index_col=0,header=0).values
    np.savetxt('LZCorr/Lgrm-'+snp,makePSD(grm),delimiter=',')
    

    ################################################### grm ###################################################

    os.symlink('../'+nameForGRM+'/ped/eigen-all','ped/eigen-all')
    os.symlink('../'+nameForGRM+'/ped/eigen-all','ped/eigen-1')
    os.symlink('../'+nameForGRM+'/ped/eigen-all','ped/eigen-2')

    return()
