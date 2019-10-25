import pickle
import os
import dropbox
import pdb

def DBRead(path,parms):    
    with open(path,'rb') as f:
        data=pickle.loads(f.read())
        
def DBWrite(data,path,parms): 
    with open(path,'wb') as f:
        f.write(pickle.dumps(data))
        
    return()
        
def DBIsFile(path,parms):
    return(os.path.exists(path))

def DBListFolder(path,parms):
    return(os.listdir(path))

def DBLog(msg,parms):
    logName=parms['logName']
    firstEntry=parms['firstEntry']
    
    append='w+' if firstEntry else 'a'
    #print(append,flush=True)   
    pdb.set_trace()
    with open(logName,append) as f:
        f.write(msg+'\n')
        
    parms['firstEntry']=False
        
    return()
    
def DBCreateFolder(path,parms):
    if DBIsFile(path,parms):
        os.rmtree(path)

    os.mkdir(path)
    
    return()
