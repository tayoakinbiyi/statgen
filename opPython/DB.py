import pickle
import os
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
        
def DBWrite(data,path,parms):
    with open(path,'wb+') as f:
        f.write(pickle.dumps(data))
    
    return()

def DBRead(path,parms):
    with open(path,'rb') as f:
        data=pickle.loads(f.read())
    
    return(data)

def DBLogHelp(local,x):
    return(os.path.getmtime('log/'+x))

def DBLogStart(arg,parms):
    local=parms['local']
    
    logOp=input('log : ""->new, 1->last :')
    if len(logOp)==0:
        logName='log/'+arg+'-'+str(datetime.now())
    else:
        partialDBLogHelp=partial(DBLogHelp,local)
        logNames=os.listdir('log')
        logNames.sort(key=partialDBLogHelp)
        logName='log/'+logNames[-1]

    parms['logName']=logName
    
    return()
    
def DBLog(msg,parms):    
    logName=parms['logName']
            
    with open(logName,'a+') as f:
        f.write(msg+'\n')
                
    return()
    
def DBCreateFolder(path,parms):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    
    return()

