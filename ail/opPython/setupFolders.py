from ail.opPython.DB import *
import os
import json
import dropbox
import pdb

def setupFolders(parms,arg):
    local=parms['local']
    name=parms['name']
    snpChr=parms['snpChr']
    
    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.chdir(local+name)
        os.mkdir('score')
        os.mkdir('fastlmm')
        os.mkdir('LZCorr')
        os.mkdir('diagnostics')
    else:
        os.chdir(local+name)    

    DBCreateFolder('holds',parms)

    DBLogStart(arg,parms)
    DBLog(json.dumps(parms,indent=3),parms)
        
    return(parms)