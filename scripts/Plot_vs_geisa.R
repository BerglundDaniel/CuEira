csvNorm=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/single_noferm/10ks_noferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csv9Stream=read.csv("/home/daniel/Project/Results/No_event/saturated/10ks_noferm_prof_1gpu_9stream_out.csv", header = TRUE, sep = ",",dec = ".")

#10ks
geisaCov=c(79.6,144.9,236.6,736.4) #10ki, 0 5 10 20 cov
geisaInd=c(24.9,79.6,1167.9) #0 cov, 2k, 10k, 100k ind

rows=nrow(csvNorm)

ncov=4
nind=4
nstream=4
ngpu=4

cov_a=c(0,5,10,20)
ind_a=c(2000,10000,100000,200000)
ind_ar=c(2000,10000,100000)

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

stream1=1 ##Actually 9 but stored in 1
stream2=4
stream3=3
stream4=2

res=array(0,dim=c(ncov,ngpu,nstream,nind))

for(row in 1:rows){
  if(csvNorm$snp[row]==10000){
    cov=csvNorm$covariate[row]
    ind=csvNorm$individuals[row]
    gpu=csvNorm$GPUs[row]
    stream=csvNorm$streams[row]
    
    cov_i=match(cov, cov_a)
    ind_i=match(ind, ind_a)
    res[cov_i,gpu,stream,ind_i]=csvNorm$cueira_calc[row]
  }
}

for(row in 1:nrow(csv9Stream)){
  cov=csv9Stream$covariate[row]
  ind=csv9Stream$individuals[row]
  gpu=csv9Stream$GPUs[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  res[cov_i,gpu,stream1,ind_i]=csv9Stream$cueira_calc[row]
}

##################Plot

#Ind
data=res[1,1:ngpu,1:nstream,1:nind]
xrange=range(ind_ar)
yrange=range(c(0,10))
  
file=paste("/home/daniel/Project/Results/geisa_saturated_speedup_ind","", sep="")
file=paste(file,".png", sep="")
png(file)
  
plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="Relative speed", cex.axis=s_ax, cex.lab=s_lab)
  
lines(ind_ar,geisaInd/data[1,stream1,1:3],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
lines(ind_ar,geisaInd/data[2,stream2,1:3],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
lines(ind_ar,geisaInd/data[3,stream3,1:3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
lines(ind_ar,geisaInd/data[4,stream4,1:3],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
title("Speed compared with GEISA",cex.main=s_title)
legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
dev.off()

#Cov
data=res[1:ncov,1:ngpu,1:nstream,1]
xrange=range(cov_a)
yrange=range(c(0,40))
  
file=paste("/home/daniel/Project/Results/geisa_saturated_speedup_cov","", sep="")
file=paste(file,".png", sep="")
png(file)
  
plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="Relative speed", cex.axis=s_ax, cex.lab=s_lab)
  
lines(cov_a,geisaCov/data[1:ncov,1,stream1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
lines(cov_a,geisaCov/data[1:ncov,2,stream2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
lines(cov_a,geisaCov/data[1:ncov,3,stream3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
lines(cov_a,geisaCov/data[1:ncov,4,stream4],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
title("Speed compared with GEISA",cex.main=s_title)
legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
dev.off()