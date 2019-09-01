import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool, cpu_count, freeze_support, set_start_method
import random
import pdb
from scipy.stats import norm, beta
import psutil
import time

from python.myStats import *
from python.ghc import *
from python.ggnull import *
from python.gbj import *

def fitStats(ggnullDat,ghcDat,sigmaHat,z,parms):
    statNames=parms['statNames']

    stat=pd.DataFrame()
    fail=pd.DataFrame()
        
    statGG,failGG=ggnull(z,ggnullDat,files)
    stat.insert(stat.shape[1],'ggnull',statGG)
    fail.insert(fail.shape[1],'ggnull',failGG)    
    print('ggnull',psutil.virtual_memory().percent)

    statGHC,failGHC=ghc(z,ghcDat,files)
    stat.insert(stat.shape[1],'ghc',statGHC)
    fail.insert(fail.shape[1],'ghc',failGHC)    
    print('ghc',psutil.virtual_memory().percent)

    statGBJ,failGBJ=gbj(z,files)
    stat.insert(stat.shape[1],'gbj',statGBJ)
    fail.insert(fail.shape[1],'gbj',failGBJ)    
    print('gbj',psutil.virtual_memory().percent)
    
    statS,failS=myStats(z,files)
    stat=pd.concat([stat,statS],axis=1)
    fail=pd.concat([fail,failS],axis=1)
    print('myStats',psutil.virtual_memory().percent)
    
    return(stat[statNames],fail[statNames])
