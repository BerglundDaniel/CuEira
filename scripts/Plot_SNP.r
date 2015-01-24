csvNorm=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/single_noferm/10ks_noferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csvSNP=read.csv("/home/daniel/Project/Results/3_4gpu_sim_100_500ks/cat_out.csv", header = TRUE, sep = ",",dec = ".")

rowsNorm=nrow(csvNorm)
rowsSNP=nrow(csvSNP)

ncov=2
nind=2
nstream=4
ngpu=4
nsnp=3

snp_a=c(10000,100000,500000)
cov_a=c(0,20)
ind_a=c(10000,100000)

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

res=array(0,dim=c(ncov,ngpu,nstream,nind,nsnp))

for(row in 1:rowsNorm){
  cov=csvNorm$covariate[row]
  ind=csvNorm$individuals[row]
  gpu=csvNorm$GPUs[row]
  stream=csvNorm$streams[row]
  snp=csvNorm$snp[row]
  
  if((cov==0||cov==20)&&(ind==10000||ind==100000)){
    cov_i=match(cov, cov_a)
    ind_i=match(ind, ind_a)
    snp_i=match(snp, snp_a)
    res[cov_i,gpu,stream,ind_i,snp_i]=csvNorm$cueira_calc[row]
  }
}

for(row in 1:rowsSNP){
  cov=csvSNP$covariate[row]
  ind=csvSNP$individuals[row]
  gpu=csvSNP$GPUs[row]
  stream=csvSNP$streams[row]
  snp=csvSNP$snp[row]
  
  cov_i=match(cov, cov_a)
  ind_i=match(ind, ind_a)
  snp_i=match(snp, snp_a)
  res[cov_i,gpu,stream,ind_i,snp_i]=csvSNP$cueira_calc[row]
}

########Plot
stream=4

for(ind in 1:nind){
  for(cov in 1:ncov){
    file=paste("/home/daniel/Project/Results/snps_cov_",cov_a[cov], sep="")
    file=paste(file,ind_a[ind], sep="_ind_")
    file=paste(file,".png", sep="")
    png(file)
    
    data=res[cov,1:ngpu,stream,ind,1:nsnp]
    
    xrange=range(snp_a)
    yrange=range(data)
    
    plot(xrange, yrange, type="n", xlab="Number of SNPs", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab)
    
    for(gpu in 1:ngpu){
      lines(snp_a,data[gpu,1:nsnp],type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
    }
    
    title("Calculation Time",cex.main=s_title)
    legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
    
    dev.off()
  }
}

part=c(10,50)
for(ind in 1:nind){
  for(cov in 1:ncov){
    file=paste("/home/daniel/Project/Results/snps_eff_cov_",cov_a[cov], sep="")
    file=paste(file,ind_a[ind], sep="_ind_")
    file=paste(file,".png", sep="")
    png(file)
    
    data=res[cov,1:ngpu,stream,ind,1:nsnp]
    
    xrange=range(snp_a)
    yrange=range(c(0,2))
    
    plot(xrange, yrange, type="n", xlab="Number of SNPs", ylab="Time relative to 10 000 SNPs", cex.axis=s_ax, cex.lab=s_lab)
    
    for(gpu in 1:ngpu){
      lines(snp_a[2:nsnp],(data[gpu,2:nsnp]/data[gpu,1])/part,type="b",lwd=2,lty=lty_a[gpu],pch=pch_a[gpu])
    }
    
    title("Calculation Time",cex.main=s_title)
    legend(xrange[1], yrange[2], c(1,2,3,4), cex=s_legend, pch=pch_a, lty=lty_a, title="GPU")
    
    dev.off()
  }
}




