def myMain(parms,mainDef):
    gbjR=importr('GBJ')
                
    diagnostics(mainDef)    
    log(parms)
    
    numTraits=parms['numTraits']
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    Y,QS,M,snps=makeSim(parms,fit=True)
    vY=np.corrcoef(Y,rowvar=False)
    plotCorr(vY,'vY')
    myHist(vY[np.triu_indices(numTraits,1)],'vY-hist')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    waldH0,etaH0=runLimix(Y,QS,M,snps)
    plotZ(waldH0,prefix='waldH0-')
    np.savetxt('waldH0',waldH0,delimiter='\t')
    
    #######################################################################################################
    
    #waldH0=np.loadtxt('waldH0',delimiter='\t')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
        
    vZ=np.corrcoef(waldH0,rowvar=False)
    plotCorr(vZ,'vZ')
    offDiag=vZ[np.triu_indices(numTraits,1)]
    
    myHist(offDiag,'vZ-hist')
    
    L=makePSD(vZ)
    stat=ELL.ell.ell(int(.3*numTraits),numTraits,offDiag=offDiag)
    z=np.matmul(norm.rvs(size=[int(1e6),ctrl['numTraits']]),L.T)
    stat.preCompute(1e3)
    stat.addRef(stat.preScore(z))
    scoreH0=stat.preScore(waldH0)
    stat.plot(stat.monteCarlo(scoreH0),'diagnostics/ellH0Dep')
    stat.plot(gbj(gbjR.GBJ,waldH0,offDiag=offDiag),'diagnostics/gbjH0Dep')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
        
    vZ=np.corrcoef(waldH0,rowvar=False)
    L=makePSD(vZ)
    offDiag=vZ[np.triu_indices(numTraits,1)]
    stat=ELL.ell.ell(int(.3*numTraits),numTraits,offDiag=offDiag)
    z=np.matmul(norm.rvs(size=[int(1e6),ctrl['numTraits']]),L.T)
    stat.addRef(stat.gnullScore(z))
    scoreH0=stat.gnullScore(waldH0)
    stat.plot(stat.monteCarlo(scoreH0),'diagnostics/ellH0DepMid')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    stat=ELL.ell.ell(int(.3*numTraits),numTraits)
    z=norm.rvs(size=[int(1e6),ctrl['numTraits']])
    stat.addRef(stat.gnullScore(z))
    scoreH0=stat.gnullScore(waldH0)
    stat.plot(stat.monteCarlo(scoreH0),'diagnostics/ellH0')
    #stat.plot(gbj(gbjR.GBJ,waldH0),'diagnostics/gbjH0')

    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    '''
    waldH1=runH1(0.1,int(parms['numTraits']*.1),waldH0,Y,QS,M,snps)
    plotZ(waldH1,prefix='waldH1-')
    np.savetxt('waldH1',waldH1,delimiter='\t')
    
    #######################################################################################################
    
    waldH1=np.loadtxt('waldH1',delimiter='\t')
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    scoreH1=stat.gnullScore(waldH1)
    stat.plot(stat.monteCarlo(scoreH1),'diagnostics/ellH1')
    stat.plot(gbj(gbjR.GBJ,waldH1),'diagnostics/gbjH1')
    '''
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    git('{} : {} Y, run with dep and indep ell'.format(sys.argv[0][:-3],parms['yParm']))
