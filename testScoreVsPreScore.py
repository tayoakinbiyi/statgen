def myMain(parms,stat,mainDef):
    gbjR=importr('GBJ')
        
    diagnostics(mainDef)    
    log(parms)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    Y,QS,M,snps=makeSim(parms,fit=True)
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
    waldH0,etaH0=runLimix(Y,QS,M,snps)
    plotZ(waldH0,prefix='waldH0-')
    np.savetxt('waldH0',waldH0,delimiter='\t')
    
    #######################################################################################################
    '''
    waldH0=np.loadtxt('waldH0',delimiter='\t')
    '''
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    
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
    
    git('{} : {} {}'.format(sys.argv[0][:-3],parms['pedigreeMult'],parms['numSubjects']))