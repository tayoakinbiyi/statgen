from ail.opPython.DB import *
import os
import json
import dropbox
import pdb

def setupFolders(parms):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']

    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.chdir(local+name)
        os.mkdir('process')
        os.mkdir('score')
        os.mkdir('geneDrop')
        os.mkdir('holds')
        os.mkdir('finished')
        os.mkdir('iidZ')
        os.mkdir('LZCorr')
        os.mkdir('fastlmm')
        
        os.mkdir('ped')
        for snp in snpChr:
            os.mkdir('ped/eigen-'+snp)

        os.mkdir('plotCorr')
        os.mkdir('minP')
        os.mkdir('log')
        os.mkdir('plotZ')
        os.mkdir('power')
        os.mkdir('pvalPlot')
    else:
        os.chdir(local+name)    

    DBLog(json.dumps(parms,indent=3),parms)
        
    return(parms)