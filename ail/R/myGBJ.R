suppressWarnings(suppressMessages(library(R.utils)))

args <- commandArgs(trailingOnly = TRUE)

fileNum=args[1]
path=args[2]

sourceDirectory(paste(path,'R/gbj',sep=''))

z=read.csv(paste(path,'gbj/z_',fileNum,'.csv',sep=''),header=FALSE)
pairwise_cors=as.numeric(read.csv(paste(path,'ebb/pairwise_cors.csv',sep=''),header=FALSE)[,1])

out=data.frame(Value=rep(NA,dim(z)[1]),Fail=rep(NA,dim(z)[1]),stringsAsFactors = FALSE)

for (row in 1:dim(z)[1]) {
  gbj=GBJ_objective(as.numeric(z[row,]),dim(z)[2],pairwise_cors=pairwise_cors)
  out[row,]=c(max(gbj),mean(gbj[1:ceiling(length(gbj)/2)]==0))
}

write.csv(out,paste(path,'gbj/gbj_',fileNum,'.csv',sep=''),row.names=FALSE)