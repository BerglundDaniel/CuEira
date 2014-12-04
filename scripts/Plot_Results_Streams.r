csv=read.csv("/home/daniel/Project/Results/10streams/100ki/10ks_noferm_prof_100ki_0cov_10streams_out.csv", header = TRUE, sep = ",",dec = ".")

rows=nrow(csv)

ncov=1
nind=1
nstream=10
ngpu=4

cov_a=c(0)
ind_a=c(10000)

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

res=array(0,dim=c(ngpu,nstream))

for(row in 1:rows){
  gpu=csv$GPUs[row]
  stream=csv$streams[row]
  
  res[gpu,stream]=csv$cueira_calc[row]
}

xrange=range(1:nstream)
yrange=range(res)

file="/home/daniel/Project/Results/gpu_10streams_10ks_100ki.png"
png(file)

plot(xrange, yrange, type="n", xlab="Number of Streams", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)

for(gpu in 1:ngpu){
  lines(1:nstream,res[gpu,1:nstream],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
}

title("Calculation Time",cex.main=s_title)
legend(8.5, yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
dev.off()

##################Time spent
cols3=c("black","gray20","gray50")
cols4=c("black","gray20","gray50","gray70")

LR_n=4
cpu_i=1
gpu_i=2
transFrom_i=3
transTo_i=4
res_LR=array(0,dim=c(ncov,ngpu,nstream,nind,LR_n))

DH_n=3
recode_i=1
read_i=2
model_i=3
res_DH=array(0,dim=c(ncov,ngpu,nstream,nind,DH_n))

for(row in 1:rows){
  cov=csv$covariate[row]
  ind=csv$individuals[row]
  gpu=csv$GPUs[row]
  stream=csv$streams[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
    
  #DH
  res_DH[cov_i,gpu,stream,ind_i, recode_i]=csv$Recode[row]
  res_DH[cov_i,gpu,stream,ind_i, read_i]=csv$ReadSNP[row]
  res_DH[cov_i,gpu,stream,ind_i, model_i]=csv$ApplyStatModel[row]
    
  #LR
  res_LR[cov_i,gpu,stream,ind_i, cpu_i]=csv$LR_CPU[row]
  res_LR[cov_i,gpu,stream,ind_i, gpu_i]=csv$LR_GPU[row]
  res_LR[cov_i,gpu,stream,ind_i, transFrom_i]=csv$LR_transferFromDevice[row]
  res_LR[cov_i,gpu,stream,ind_i, transTo_i]=csv$LR_transferToDevice[row]+csv$LR_Config_transferToDevice[row]
}

#Stream
#DH
for(gpu in 1:ngpu){
  cm=array(0,dim=c(DH_n,nstream))
  
  file=paste("/home/daniel/Project/Results/timedist_DH_10streams_10ks_100ki_gpu",gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(stream in 1:nstream){
    t=0
    for(dh in 1:DH_n){
      t=t+res_DH[1,gpu,stream,1,dh]
    }
    for(dh in 1:DH_n){
      cm[dh,stream]=res_DH[1,gpu,stream,1,dh]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="DataHandler Time", xlab="Number of Streams", ylab="Fraction of time spent", col=cols3, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("Recode","Read","ApplyStatisticModel"), cex=s_legend, bty="n", fill=cols3)
  
  dev.off()
}

#LR
for(gpu in 1:ngpu){
  cm=array(0,dim=c(LR_n,nstream))
  
  file=paste("/home/daniel/Project/Results/timedist_LR_10streams_10ks_100ki_gpu",gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(stream in 1:nstream){
    t=0
    for(lr in 1:LR_n){
      t=t+res_LR[1,gpu,stream,1,lr]
    }
    for(lr in 1:LR_n){
      cm[lr,stream]=res_LR[1,gpu,stream,1,lr]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="Logistic Regression Time", xlab="Number of Streams", ylab="Fraction of time spent", col=cols4, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("CPU","GPU","transfer from GPU","transfer to GPU"), cex=s_legend, bty="n", fill=cols4)
  
  dev.off()
}


#####5+stream
lower=5
xrange=range(lower:nstream)
yrange=range(res[1:ngpu,lower:nstream])

file="/home/daniel/Project/Results/gpu_5to10streams_10ks_100ki.png"
png(file)

plot(xrange, yrange, type="n", xlab="Number of Streams", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)

for(gpu in 1:ngpu){
  lines(lower:nstream,res[gpu,lower:nstream],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
}

title("Calculation Time",cex.main=s_title)
legend(8.5, yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
dev.off()