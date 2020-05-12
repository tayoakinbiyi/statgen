#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
library(GBJ)

file=args[1]
func=args[2]

offDiag=read.table('offDiag',header=F,sep='\t')[,1]
z=read.table(file,header=F,sep='\t')
ans=array(dim=c(dim(z)[1],1))

for(row in 1:dim(z)[1]) {
    print(paste(func,row,sep=' '))
    ans[row]=do.call(func,list(test_stats=as.double(z[row,]), pairwise_cors = offDiag))[paste(func,'_pvalue',sep='')]
}

write.table(ans,file,col.names=F,row.names=F,quote=F,sep='\t')
    