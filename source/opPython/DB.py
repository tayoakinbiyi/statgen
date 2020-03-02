import os
import pdb
import subprocess
from dill.source import getsource
import json
import re
import pprint

def DBFinish(local,mainDef):
    nm=local+'archive/'
    
    subprocess.call(['rm','-rf',nm])
    subprocess.call(['cp','-rf','source',nm])
    with open(nm+'main','w+') as f:
        f.write(mainDef)
    subprocess.call(['cp','diagnostics/log',nm])
    subprocess.call(['cp','-r','diagnostics',nm])  
    
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

