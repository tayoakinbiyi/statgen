import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from python.ggof import *
from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *

def monteCarlo(L,sigName,eps,mu,Reps,ebb,var):
    N=len(L)
    
    z=np.matmul(L.T,np.random.normal(0,1,size=(N,Reps))).T
    
    if mu*eps>0:
        z[:,range(eps)]+=mu
    
    power=pd.DataFrame()
    fail=pd.DataFrame()
        
    print(eps,mu,psutil.virtual_memory().percent)

    powerGG,failGG=ggnull(z,sigName,ebb)
    power=power.append(powerGG)
    fail=fail.append(failGG)    
    print('ggnull',psutil.virtual_memory().percent)

    powerGHC,failGHC=ghc(z,sigName,var)
    power=power.append(powerGHC)
    fail=fail.append(failGHC)    
    print('ghc',psutil.virtual_memory().percent)

    powerGBJ,failGBJ=gbj(z,sigName)
    power=power.append(powerGBJ)
    fail=fail.append(failGBJ)    
    print('gbj',psutil.virtual_memory().percent)
    
    power=power.append(myStats(z))
    print('myStats',psutil.virtual_memory().percent)
    
    power.index=len(power)*[0]
    power=power.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)
    
    if len(fail)>0:
        fail.index=len(fail)*[0]
        fail=fail.merge(pd.DataFrame([[eps,mu]],columns=['eps','mu'],index=[0]),left_index=True,right_index=True)

    return(power,fail)

