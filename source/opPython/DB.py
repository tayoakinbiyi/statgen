import pickle
import os
import pdb
from datetime import datetime
from functools import partial
import shutil
import subprocess

def DBWipe(path,parms):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
        
def DBWrite(data,path,parms):
    with open(path,'wb+') as f:
        f.write(pickle.dumps(data))
    
    return()

def DBRead(path,parms):
    with open(path,'rb') as f:
        data=pickle.loads(f.read())
    
    return(data)

def DBLogStart(parms):
    local=parms['local']
        
    subprocess.call(['tar','-zcf','source.tar.gz',local+'source'])
    subprocess.call(['cp',local+'source/main/'+parms['file'],parms['file']])
    
    return()

def DBFinish(parms):
    local=parms['local']
    
    nm=local+'archive/'+parms['name']+'-'+str(datetime.now())
    subprocess.call(['mkdir',nm])
    subprocess.call(['cp',parms['file'],nm])
    subprocess.call(['cp','source.tar.gz',nm])
    subprocess.call(['cp','log',nm])
    subprocess.call(['cp','-r','diagnostics',nm])   
    
    return()
    
def DBLog(msg): 
    print(msg,flush=True)
    
    with open('log','a+') as f:
        f.write(msg+'\n')
                
    return()
    
def DBCreateFolder(path,parms):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    return()

