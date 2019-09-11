from ail.opPython.DB import *
import os

def setupFolders(parms):
    local=parms['local']
    name=parms['name']
    dbToken=parms['dbToken']

    if not DBIsFile('',name[:-1],parms):
        DBCreateFolder(name[:-1],parms)
        DBCreateFolder(name+'process',parms)
        DBCreateFolder(name+'score',parms)
        DBCreateFolder(name+'sim',parms)
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
        os.mkdir(local+name+'sim')
        os.mkdir(local+name+'usThem')
        os.mkdir(local+name+'qq')
        os.mkdir(local+name+'corr')
        os.mkdir(local+name+'z2')
        os.mkdir(local+name+'man')
        os.mkdir(local+name+'plots')

    DBSyncLocal('data',parms)