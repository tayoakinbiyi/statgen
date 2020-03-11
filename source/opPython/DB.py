import os
import pdb
import subprocess
from dill.source import getsource
import json
import re
import pprint

def diagnostics(mainDef,parms):
    subprocess.call(['rm','-rf','diagnostics/'])
    subprocess.call(['mkdir','diagnostics'])
    with open('diagnostics/main','w+') as f:
        f.write(mainDef)
    DBLog(parms)
    
    return()
    
    
def git(local):
    nm=local+'archive/'
    
    subprocess.call(['rm','-rf',nm])
    subprocess.call(['cp','-rf','source',nm])
    subprocess.call(['cp','-r','diagnostics',nm])  
    subprocess.call(['cp','diagnostics/main',nm])  
    
    return()
    
def DBLog(msg): 
    print(msg,flush=True)

    with open('diagnostics/log','a+') as f:
        f.write(pprint.pformat(msg,compact=True)+'\n')
                
    return()
    
def DBCreateFolder(path,parms):
    if os.path.exists(path):
        subprocess.call(['rm','-rf',path])

    os.mkdir(path)
    
    return()

