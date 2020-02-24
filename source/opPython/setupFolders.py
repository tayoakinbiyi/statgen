from opPython.DB import *
import os
import json
import pdb
import subprocess
from opPython.MyEncoder import *

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
        os.mkdir('power')
        os.mkdir('inputs')
    else:
        os.chdir(local+name)    
    
    DBLogStart(parms)
    DBLog(json.dumps(ctrl, cls=MyEncoder, sort_keys=True, indent=3))
        
    return(parms)