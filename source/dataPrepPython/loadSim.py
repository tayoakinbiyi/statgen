    Y=np.loadtxt('Y',delimiter='\t')
    Y=Y.reshape(len(Y),-1)

    Q0=np.loadtxt('Q0',delimiter='\t')
    Q0=Q0.reshape(len(Q0),-1)
    Q1=np.loadtxt('Q1',delimiter='\t')
    Q1=Q1.reshape(len(Q1),-1)
    S0=np.loadtxt('S0',delimiter='\t')
    QS=((Q0, Q1), S0)

    M=np.loadtxt('M',delimiter='\t')
    M=M.reshape(len(M),-1)

    snps={'data':np.loadtxt('snps',delimiter='\t')}
