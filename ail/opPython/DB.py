import pickle
import os
import dropbox
import pdb
from datetime import datetime
from functools import partial
import shutil

def DBWipe(path,parms):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
        
def DBLogHelp(local,x):
    return(os.path.getmtime(local+'log/'+x))

def DBLogStart(arg,parms):
    local=parms['local']
    
    logOp=input('log : ""->new, 1->last :')
    if len(logOp)==0:
        logName=local+'log/'+arg+'-'+str(datetime.now())
    else:
        partialDBLogHelp=partial(DBLogHelp,local)
        logNames=os.listdir(local+'log')
        logNames.sort(key=partialDBLogHelp)
        logName=local+'log/'+logNames[-1]

    parms['logName']=logName
    
    return()
    
def DBLog(msg,parms):
    local=parms['local']
    
    logName=parms['logName']
            
    with open(logName,'a+') as f:
        f.write(msg+'\n')
                
    return()
    
def DBCreateFolder(path,parms):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    return()

