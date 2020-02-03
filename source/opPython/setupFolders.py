from opPython.DB import *
import os
import json
import pdb
import subprocess
import numpy as np

def setupFolders(ctrl,ops):
    parms={**ctrl,**ops}

    parms['local']=os.getcwd()+'/'
    local=parms['local']
    
    parms['name']=parms['file'][:-3]
    name=parms['name']
    
    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.chdir(local+name)
        os.mkdir('score')
        os.mkdir('LZCorr')
        os.mkdir('diagnostics')
        os.mkdir('grm')
        os.mkdir('output')
        os.mkdir('pvals')
        os.mkdir('ref')
        os.mkdir('power')
        os.mkdir('stats')
        os.mkdir('holds')
    else:
        os.chdir(local+name)    
    
    DBLogStart(parms)
    DBLog(json.dumps(ctrl,indent=3))
    DBLog(json.dumps(ops,indent=3))
        
    return(parms)