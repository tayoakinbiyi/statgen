import os
import pdb
import subprocess
from dill.source import getsource
    

def DBFinish(local,mainDef):
    nm=local+'archive/'
    
    subprocess.call(['rm','-rf',nm])
    subprocess.call(['cp','-rf','source',nm])
    with open(nm+'main','w+') as f:
        f.write(mainDef)
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
