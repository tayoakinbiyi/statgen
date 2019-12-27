from ail.opPython.DB import *
import os
import json
import pdb

def setupFolders(parms,arg):
    local=parms['local']
    name=parms['name']
    
    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.chdir(local+name)
        os.mkdir('score')
        os.mkdir('LZCorr')
        os.mkdir('diagnostics')
        os.mkdir('grm')
        os.mkdir('output')
        os.mkdir('log')
        os.mkdir('pvals')
        os.mkdir('ref')
        os.mkdir('power')
        os.mkdir('stats')
        os.mkdir('holds')
    else:
        os.chdir(local+name)    

    DBLogStart(arg,parms)
    DBLog(json.dumps(parms,indent=3),parms)
        
    return(parms)