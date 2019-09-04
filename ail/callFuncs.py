import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import pdb
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from multiprocessing import cpu_count
import warnings
#warnings.simplefilter("error")

from ail.dataPrepPython.process import *
from ail.dataPrepPython.genCorr import *
from ail.dataPrepPython.genCorrMats import *
from ail.dataPrepPython.genMeans import *
from ail.dataPrepPython.score import *

from ail.plotPython.manhattanPlots import *
from ail.plotPython.plotCorr import *
from ail.plotPython.qqPlots import *
from ail.plotPython.usThem import *
from ail.plotPython.zSquaredHists import *


def callFuncs(parms):
    local=parms['local']
    name=parms['name']
    dbToken=parms['dbToken']

    if not DBIsFile('',name[:-1],parms):
        DBCreateFolder(name[:-1],parms)
        DBCreateFolder(name+'process',parms)
        DBCreateFolder(name+'score',parms)
        DBCreateFolder(name+'usThem',parms)
        DBCreateFolder(name+'qq',parms)
        DBCreateFolder(name+'corr',parms)
        DBCreateFolder(name+'z2',parms)
        DBCreateFolder(name+'man',parms)
        DBCreateFolder(name+'plots',parms)

    if not os.path.exists(local+name):
        os.mkdir(local+name)
        os.mkdir(local+name+'process')
        os.mkdir(local+name+'score')
        os.mkdir(local+name+'usThem')
        os.mkdir(local+name+'qq')
        os.mkdir(local+name+'corr')
        os.mkdir(local+name+'z2')
        os.mkdir(local+name+'man')
        os.mkdir(local+name+'plots')

    DBSyncLocal('data',parms)

    if parms['process']:
        print('process')
        process(parms)

    # create the actual Z scores by running gemma lmm
    if parms['score']:
        print('gen scores')   
        score(parms)

    if parms['corr']:
        print('genMeans')
        genMeans(parms)
        print('genCorrMats')
        genCorrMats(parms)
        print('makeCorrPlots')
        plotCorr(parms)

    # qq plots of p-vals
    if parms['qq']:
        print('qq plots')
        qqPlots(parms)

    # plot histogram of squared Z scores by chromosome
    if parms['z2']:
        print('do zvar')   
        zSquaredHists(parms)

    if parms['man']:    
        print('MA Plots')
        manhattanPlots(parms,B=10)

    if parms['usThem']:
        print('us Them')
        usThem({**parms,'cpu':10})
