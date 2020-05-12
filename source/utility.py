import os
import json
import pdb
import subprocess
from zipfile import ZipFile
import pprint
import shutil
import sys
import numpy as np
import socket
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from scipy.stats import beta

from plotPython.myQQ import *

def setupFolders():
    name=sys.argv[0][:-3]
    
    if not os.path.exists(name):
        os.mkdir(name)
        os.chdir(name)
    else:
        os.chdir(name)    
           
    return()

def diagnostics(seed=None):
    subprocess.call(['rm','-rf','diagnostics/'])
    subprocess.call(['mkdir','diagnostics'])
    shutil.make_archive('diagnostics/python', "zip", '../source')
    subprocess.call(['cp','../'+sys.argv[0],'diagnostics/'+sys.argv[0]])
    
    if seed is None:
        seed=int(np.random.randint(1e6))
        
    np.random.seed(seed)
    log('random seed {}'.format(seed))

    return()
       
def git(msg): 
    hostname=socket.gethostname()
        
    local=os.getcwd()+'/'
    os.chdir('../'+hostname)
    subprocess.call(['git','fetch','--all'])
    subprocess.call(['git','reset','--hard','HEAD'])
    for file in [file for file in os.listdir() if file!='.git']:
        subprocess.call(['git','rm','-r',file])
    for file in os.listdir(local+'diagnostics'):
        if os.stat(local+'diagnostics/'+file).st_size<50*1024**2:
            subprocess.call(['cp','-rf',local+'diagnostics/'+file,file])  
    subprocess.call(['git','add','-A'])
    subprocess.call(['git','commit','-m',msg])
    subprocess.call(['git','push','-f','origin','archive'])
    os.chdir(local)
    
    return()
    
def log(msg): 
    print(msg,flush=True)

    with open('diagnostics/log','a+') as f:
        f.write(pprint.pformat(msg,compact=True)+'\n')
                
    return()

def makeL(df):
    U,D,Vt=np.linalg.svd(df)
    L=np.matmul(U,np.diag(np.sqrt(D)))

    return(L)

