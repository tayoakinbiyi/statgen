from opPython.DB import *
import os
import json
import pdb
import subprocess

def setupFolders(ctrl,ops):
    parms={**ctrl,**ops}

    parms['local']=os.getcwd()+'/'
    local=parms['local']
    
    DBCreateFolder('output',parms)

    parms['name']=parms['file'][:-3]
    name=parms['name']
    
    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.chdir(local+name)
        os.mkdir('score')
        os.mkdir('grm')
        os.mkdir('output')
        os.mkdir('snps')
        os.mkdir('Y')
        os.mkdir('cov')
    else:
        os.chdir(local+name)    
    
    subprocess.call(['cp','-rLf',local+'source','.'])
        
    return(parms)