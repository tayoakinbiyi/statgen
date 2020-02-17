import os
import pdb
from datetime import datetime
from functools import partial
import shutil
import subprocess

def DBLogStart(parms):
    local=parms['local']
        
    subprocess.call(['cp','-rLf',local+'source','.'])
    subprocess.call(['cp',local+'source/main/'+parms['file'],'diagnostics/'+parms['file']])
    subprocess.call(['rm','-f','log'])
    
    return()

def DBFinish(parms):
    local=parms['local']
    
    nm=local+'archive/'
    
    subprocess.call(['rm','-rf',nm])
    subprocess.call(['cp','-rf','source',nm])
    subprocess.call(['cp',parms['file'],nm+parms['name']])
    subprocess.call(['cp','log',nm])
    subprocess.call(['cp','-rf','diagnostics',nm])  
    
    return()
    
def DBLog(msg): 
    print(msg,flush=True)
    
    with open('log','a+') as f:
        f.write(msg+'\n')
                
    return()
    
def DBCreateFolder(path,parms):
    if os.path.exists(path):
        subprocess.call(['rm','-rf',path])

    os.mkdir(path)
    
    return()

