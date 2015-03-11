csvNorm=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/single_noferm/10ks_noferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csvFerm=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/ferm/10ks_ferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csvDouble=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/double/10ks_noferm_prof_double_out.csv", header = TRUE, sep = ",",dec = ".")
csvNoProf=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/noprof/10ks_noferm_noprof_out.csv", header = TRUE, sep = ",",dec = ".")
csvOptAll=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/opt_all/10ks_noferm_prof_o3_noali_fastmath_out.csv", header = TRUE, sep = ",",dec = ".")
csv9Stream=read.csv("/home/daniel/Project/Results/No_event/saturated/10ks_noferm_prof_1gpu_9stream_out.csv", header = TRUE, sep = ",",dec = ".")
csvFerm9Stream=read.csv("/home/daniel/Project/Results/No_event/4gpu_4stream_sim/ferm/10ks_fix_ferm_single_out.csv", header = TRUE, sep = "\t",dec = ".")

rows=nrow(csvNorm)

ncov=4
nind=4
nstream=4
ngpu=4

cov_a=c(0,5,10,20)
ind_a=c(2000,10000,100000,200000)

stream1=1 ##Actually 9 but stored in 1
stream2=4
stream3=3
stream4=2

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

resNorm=array(0,dim=c(ncov,ngpu,nstream,nind))
resFerm=array(0,dim=c(ncov,ngpu,nstream,nind))
resDouble=array(0,dim=c(ncov,ngpu,nstream,nind))
resNoProf=array(0,dim=c(ncov,ngpu,nstream,nind))
resOptAll=array(0,dim=c(ncov,ngpu,nstream,nind))

for(row in 1:rows){
  #Norm
  cov=csvNorm$covariate[row]
  ind=csvNorm$individuals[row]
  gpu=csvNorm$GPUs[row]
  stream=csvNorm$streams[row]
    
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resNorm[cov_i,gpu,stream,ind_i]=csvNorm$cueira_calc[row]
  
  #Ferm
  cov=csvFerm$covariate[row]
  ind=csvFerm$individuals[row]
  gpu=csvFerm$GPUs[row]
  stream=csvFerm$streams[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resFerm[cov_i,gpu,stream,ind_i]=csvFerm$cueira_calc[row]
  
  #Double
  cov=csvDouble$covariate[row]
  ind=csvDouble$individuals[row]
  gpu=csvDouble$GPUs[row]
  stream=csvDouble$streams[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resDouble[cov_i,gpu,stream,ind_i]=csvDouble$cueira_calc[row]
  
  #NoProf
  cov=csvNoProf$covariate[row]
  ind=csvNoProf$individuals[row]
  gpu=csvNoProf$GPUs[row]
  stream=csvNoProf$streams[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resNoProf[cov_i,gpu,stream,ind_i]=csvNoProf$FamReaderRead[row] #Because it lacks columns
  
  #OptAll
  cov=csvOptAll$covariate[row]
  ind=csvOptAll$individuals[row]
  gpu=csvOptAll$GPUs[row]
  stream=csvOptAll$streams[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resOptAll[cov_i,gpu,stream,ind_i]=csvOptAll$cueira_calc[row]
}

#####Put in 9 stream
for(row in 1:nrow(csv9Stream)){
  cov=csv9Stream$covariate[row]
  ind=csv9Stream$individuals[row]
  gpu=csv9Stream$GPUs[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resNorm[cov_i,gpu,stream1,ind_i]=csv9Stream$cueira_calc[row]
}

#Ferm 9
for(row in 1:nrow(csvFerm9Stream)){
  cov=csvFerm9Stream$covariate[row]
  ind=csvFerm9Stream$individuals[row]
  gpu=csvFerm9Stream$GPUs[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  resFerm[cov_i,gpu,stream1,ind_i]=csvFerm9Stream$cueira_calc[row]
}

####Vs Ferm

#Ind
for(cov in 1:ncov){
  data=resNorm[cov,1:ngpu,1:nstream,1:nind]/resFerm[cov,1:ngpu,1:nstream,1:nind]
  xrange=range(ind_a)
  #yrange=range(data[1,stream1,1:nind],data[2,stream2,1:nind],data[3,stream3,1:nind],data[4,stream4,1:nind])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/sync_comp_ind_cov_",cov_a[cov], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="Less synchronisation/always synchronisation", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(ind_a,data[1,stream1,1:nind],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(ind_a,data[2,stream2,1:nind],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(ind_a,data[3,stream3,1:nind],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(ind_a,data[4,stream4,1:nind],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

#Cov
for(ind in 1:nind){
  data=resNorm[1:ncov,1:ngpu,1:nstream,ind]/resFerm[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(cov_a)
  #yrange=range(data[1:ncov,1,stream1],data[1:ncov,2,stream2],data[1:ncov,3,stream3],data[1:ncov,4,stream4])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/sync_comp_cov_ind_",ind_a[ind], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="Less synchronisation/always synchronisation", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(cov_a,data[1:ncov,1,stream1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(cov_a,data[1:ncov,2,stream2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(cov_a,data[1:ncov,3,stream3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(cov_a,data[1:ncov,4,stream4],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

####VS Double
#Ind
for(cov in 1:ncov){
  data=resNorm[cov,1:ngpu,1:nstream,1:nind]/resDouble[cov,1:ngpu,1:nstream,1:nind]
  xrange=range(ind_a)
  #yrange=range(data[1,stream1,1:nind],data[2,stream2,1:nind],data[3,stream3,1:nind],data[4,stream4,1:nind])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/double_comp_ind_cov_",cov_a[cov], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="Single precision/double precision", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(ind_a,data[1,stream1,1:nind],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(ind_a,data[2,stream2,1:nind],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(ind_a,data[3,stream3,1:nind],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(ind_a,data[4,stream4,1:nind],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

#Cov
for(ind in 1:nind){
  data=resNorm[1:ncov,1:ngpu,1:nstream,ind]/resDouble[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(cov_a)
  #yrange=range(data[1:ncov,1,stream1],data[1:ncov,2,stream2],data[1:ncov,3,stream3],data[1:ncov,4,stream4])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/double_comp_cov_ind_",ind_a[ind], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="Single precision/double precision", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(cov_a,data[1:ncov,1,stream1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(cov_a,data[1:ncov,2,stream2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(cov_a,data[1:ncov,3,stream3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(cov_a,data[1:ncov,4,stream4],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

##Vs prof
#Ind
for(cov in 1:ncov){
  data=resNorm[cov,1:ngpu,1:nstream,1:nind]/resNoProf[cov,1:ngpu,1:nstream,1:nind]
  xrange=range(ind_a)
  #yrange=range(data[1,stream1,1:nind],data[2,stream2,1:nind],data[3,stream3,1:nind],data[4,stream4,1:nind])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/prof_comp_ind_cov_",cov_a[cov], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="With profiling/withot profiling", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(ind_a,data[1,stream1,1:nind],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(ind_a,data[2,stream2,1:nind],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(ind_a,data[3,stream3,1:nind],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(ind_a,data[4,stream4,1:nind],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

#Cov
for(ind in 1:nind){
  data=resNorm[1:ncov,1:ngpu,1:nstream,ind]/resNoProf[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(cov_a)
  #yrange=range(data[1:ncov,1,stream1],data[1:ncov,2,stream2],data[1:ncov,3,stream3],data[1:ncov,4,stream4])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/prof_comp_cov_ind_",ind_a[ind], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="With profiling/withot profiling", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(cov_a,data[1:ncov,1,stream1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(cov_a,data[1:ncov,2,stream2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(cov_a,data[1:ncov,3,stream3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(cov_a,data[1:ncov,4,stream4],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

##Vs opt all
#Ind
for(cov in 1:ncov){
  data=resNorm[cov,1:ngpu,1:nstream,1:nind]/resOptAll[cov,1:ngpu,1:nstream,1:nind]
  xrange=range(ind_a)
  #yrange=range(data[1,stream1,1:nind],data[2,stream2,1:nind],data[3,stream3,1:nind],data[4,stream4,1:nind])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/opt_comp_ind_cov_",cov_a[cov], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="Without/with compiler options", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(ind_a,data[1,stream1,1:nind],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(ind_a,data[2,stream2,1:nind],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(ind_a,data[3,stream3,1:nind],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(ind_a,data[4,stream4,1:nind],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}

#Cov
for(ind in 1:nind){
  data=resNorm[1:ncov,1:ngpu,1:nstream,ind]/resOptAll[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(cov_a)
  #yrange=range(data[1:ncov,1,stream1],data[1:ncov,2,stream2],data[1:ncov,3,stream3],data[1:ncov,4,stream4])
  yrange=c(0,1.5)
  
  file=paste("/home/daniel/Project/Results/opt_comp_cov_ind_",ind_a[ind], sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="Without/with compiler options", cex.axis=s_ax, cex.lab=s_lab)
  
  lines(cov_a,data[1:ncov,1,stream1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[1])
  lines(cov_a,data[1:ncov,2,stream2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[2])
  lines(cov_a,data[1:ncov,3,stream3],type="b",lwd=2,lty=lty_a[3],pch=pch_a[3])
  lines(cov_a,data[1:ncov,4,stream4],type="b",lwd=2,lty=lty_a[4],pch=pch_a[4])
  
  title("Relative Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[1]+0.45, c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  
  dev.off()
}