import os
from ail.opPython.DB import *
from ail.genPython.makePSD import *
from ail.opPython.setupFolders import *
from ail.plotPython.plotCorr import *
import pdb
import matplotlib.pyplot as plt

parms={
    'local':os.getcwd()+'/',
    'name':'testGRM/',
    'dbToken':'YIjLc0Jkc2QAAAAAAAAELhNPLYwqK53qaNPZqrkPIgHhe6n--GwXZbmgkQwbOQMo'
}
local=parms['local']
name=parms['name']
'''
td=DBRead('natalia/process/traitData',parms)
natalia=DBRead('natalia/score/p-chr18-chr1',parms)
nataliaFullGRM=DBRead('nataliaFullGRM/score/p-chr18-chr1',parms)
xx=np.concatenate([natalia[:,19].reshape(-1,1),nataliaFullGRM[:,19].reshape(-1,1)],axis=1)
pdb.set_trace()
'''
'''
setupFolders(parms)
LNatalia=makePSD(np.loadtxt('/deeplearning/akinbiyi/natalia/process/grm-chr1.txt',delimiter='\t'))
LNataliaFullGRM=makePSD(np.loadtxt('/deeplearning/akinbiyi/nataliaFullGRM/process/grm-chr1.txt',delimiter='\t'))
DBWrite(LNatalia,name+'process/LNatalia',parms,toPickle=True)
DBWrite(LNataliaFullGRM,name+'process/LNataliaFullGRM',parms,toPickle=True)

_,__=plotCorr({'subset first':'process/LNatalia','Jsubset after':'process/LNataliaFullGRM'},'GRM Plot',parms)
'''
fig,axs=plt.subplots(1,1,dpi=150)
natalia=-np.log10(DBRead('natalia/score/p-chr2-chr1',parms,toPickle=True).flatten())
nataliaFullGRM=-np.log10(DBRead('nataliaFullGRM/score/p-chr2-chr1',parms,toPickle=True).flatten())
axs.scatter(natalia,nataliaFullGRM,marker='.',s=.01)
axs.set_xlabel('subsetFirst')
axs.set_ylabel('subsetSecond')
fig.savefig('/deeplearning/akinbiyi/testGRM/plots/chr2-chr1.png')
DBUpload('testGRM/plots/chr2-chr1.png',parms,toPickle=False)
