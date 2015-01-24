
csv=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/single_noferm/10ks_noferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
#csv=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/double/10ks_noferm_prof_double_out.csv", header = TRUE, sep = ",",dec = ".")
#csv=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/ferm/10ks_ferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")

rows=nrow(csv)

ncov=4
nind=4
nstream=4
ngpu=4

cov_a=c(0,5,10,20)
ind_a=c(2000,10000,100000,200000)

res=array(0,dim=c(ncov,ngpu,nstream,nind))

for(row in 1:rows){
  if(csv$snp[row]==10000){
    cov=csv$covariate[row]
    ind=csv$individuals[row]
    gpu=csv$GPUs[row]
    stream=csv$streams[row]
    
    cov_i=match(cov, cov_a)
    ind_i=match(ind, ind_a)
    res[cov_i,gpu,stream,ind_i]=csv$cueira_calc[row]
  }
}

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

#Individuals
for(cov in 1:ncov){
  data=res[cov,1:ngpu,1:4,1:nind]
  xrange=range(ind_a)
  
  for(stream in 1:nstream){
  data_stream=data[1:ngpu,stream,1:nind]
  yrange=range(data_stream)
    
  file=paste("/home/daniel/Project/Results/individuals_cov_",cov_a[cov], sep="")
  file=paste(file,stream, sep="_stream_")
  file=paste(file,".png", sep="")
  png(file)

  plot(xrange, yrange, type="n", xlab="Number of Individuals", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)

  for(gpu in 1:ngpu){
    lines(ind_a,data_stream[gpu,1:nind],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
  }

  title("Execution Time",cex.main=s_title)
  legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  dev.off()
  }
}

#Streams
for(ind in 1:nind){
  data=res[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(1:nstream)
  
  for(cov in 1:ncov){
  data_cov=data[cov,1:ngpu,1:nstream]
  yrange=range(data_cov)
  
  file=paste("/home/daniel/Project/Results/streams_individuals_",ind_a[ind], sep="")
  file=paste(file,cov_a[cov], sep="_cov_")
  file=paste(file,".png", sep="")
  png(file)

  plot(xrange, yrange, type="n", xlab="Number of Streams", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)

  for(gpu in 1:ngpu){
    lines(1:nstream,data_cov[gpu,1:nstream],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
  }

  title("Calculation Time",cex.main=s_title)
  legend(3.5, yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  dev.off()
}
}

#Covariates
for(ind in 1:nind){
  data=res[1:ncov,1:ngpu,1:nstream,ind]
  xrange=range(cov_a)
  yrange=range(data)
  
for(stream in 1:nstream){
  data_stream=data[1:ncov,1:ngpu,stream]
  
  file=paste("/home/daniel/Project/Results/cov_individuals",ind_a[ind], sep="")
  file=paste(file,stream, sep="_stream_")
  file=paste(file,".png", sep="")
  png(file)
  
  plot(xrange, yrange, type="n", xlab="Number of Covariates", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)
  
  for(gpu in 1:ngpu){
    lines(cov_a,data_stream[1:ncov,gpu],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
  }
  
  title("Calculation Time",cex.main=s_title)
  legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
  dev.off()
}
}

#SNP
#TODO

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
  if(csv$snp[row]==10000){
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
}

#Cov#######################
#DH
stream=3
for(ind in 1:nind){
for(gpu in 1:ngpu){
  cm=array(0,dim=c(DH_n,ncov))
  
  file=paste("/home/daniel/Project/Results/timedist_DH_cov_10ks_",ind_a[ind], sep="")
  file=paste(file,"i_gpu", sep="")
  file=paste(file,gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(cov in 1:ncov){
    t=0
    for(dh in 1:DH_n){
      t=t+res_DH[cov,gpu,stream,ind,dh]
    }
    for(dh in 1:DH_n){
      cm[dh,cov]=res_DH[cov,gpu,stream,ind,dh]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="DataHandler Time", xlab="Number of Covariates", ylab="Fraction of time spent", col=cols3, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("Recode","Read","ApplyStatisticModel"), cex=s_legend, bty="n", fill=cols3)
  
  text=c("0","5", "10", "20")
  mtext(text,1,line=1,at=c(2.5,6.5,10.5,14.5))
  
  dev.off()
}

#LR
for(gpu in 1:ngpu){
  cm=array(0,dim=c(LR_n,ncov))
  
  file=paste("/home/daniel/Project/Results/timedist_LR_cov_10ks_",ind_a[ind], sep="")
  file=paste(file,"i_gpu", sep="")
  file=paste(file,gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(cov in 1:ncov){
    t=0
    for(lr in 1:LR_n){
      t=t+res_LR[cov,gpu,stream,ind,lr]
    }
    for(lr in 1:LR_n){
      cm[lr,cov]=res_LR[cov,gpu,stream,ind,lr]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="Logistic Regression Time", xlab="Number of Covariates", ylab="Fraction of time spent", col=cols4, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("CPU","GPU","transfer from GPU","transfer to GPU"), cex=s_legend, bty="n", fill=cols4)
  
  text=c("0","5", "10", "20")
  mtext(text,1,line=1,at=c(3,8,13,18))
  
  dev.off()
}
}

#Ind###########################
#DH
stream=3
for(cov in 1:ncov){
for(gpu in 1:ngpu){
  cm=array(0,dim=c(DH_n,nind))
  
  file=paste("/home/daniel/Project/Results/timedist_DH_ind_10ks_",cov_a[cov], sep="")
  file=paste(file,"cov_gpu", sep="")
  file=paste(file,gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(ind in 1:nind){
    t=0
    for(dh in 1:DH_n){
      t=t+res_DH[cov,gpu,stream,ind,dh]
    }
    for(dh in 1:DH_n){
      cm[dh,ind]=res_DH[cov,gpu,stream,ind,dh]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="DataHandler Time", xlab="Number of Individuals", ylab="Fraction of time spent", col=cols3, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("Recode","Read","ApplyStatisticModel"), cex=s_legend, bty="n", fill=cols3)
  
  text=c("2000","10000", "100000", "200000")
  mtext(text,1,line=1,at=c(2.5,6.5,10.5,14.5))
  
  dev.off()
}

#LR
for(gpu in 1:ngpu){
  cm=array(0,dim=c(LR_n,nind))
  
  file=paste("/home/daniel/Project/Results/timedist_LR_ind_10ks_",cov_a[cov], sep="")
  file=paste(file,"cov_gpu", sep="")
  file=paste(file,gpu, sep="")
  file=paste(file,".png", sep="")
  png(file)
  
  for(ind in 1:nind){
    t=0
    for(lr in 1:LR_n){
      t=t+res_LR[cov,gpu,stream,ind,lr]
    }
    for(lr in 1:LR_n){
      cm[lr,ind]=res_LR[cov,gpu,stream,ind,lr]/t
    }
  }
  
  barplot(ylim=c(0,1),cm, main="Logistic Regression Time", xlab="Number of Individuals", ylab="Fraction of time spent", col=cols4, beside=TRUE, cex.axis=s_ax, cex.lab=s_lab)
  legend("topleft", c("CPU","GPU","transfer from GPU","transfer to GPU"), cex=s_legend, bty="n", fill=cols4)
  
  text=c("2000","10000", "100000", "200000")
  mtext(text,1,line=1,at=c(3,8,13,18))
  
  dev.off()
}
}

