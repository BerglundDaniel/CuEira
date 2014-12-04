csvNorm=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/single_noferm/10ks_noferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csvFerm=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/ferm/10ks_ferm_prof_out.csv", header = TRUE, sep = ",",dec = ".")
csvDouble=read.csv("/home/daniel/Project/Results/4gpu_4stream_sim/double/10ks_noferm_prof_double_out.csv", header = TRUE, sep = ",",dec = ".")

rows=nrow(csvNorm)

ncov=4
nind=4
nstream=4
ngpu=4

cov_a=c(0,5,10,20)
ind_a=c(2000,10000,100000,200000)


lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

index=with(csvNorm,order(csvNorm$individuals,csvNorm$covariate,csvNorm$GPUs,csvNorm$streams))
csvNormSort=csvNorm[index,]

index=with(csvFerm,order(csvFerm$individuals,csvFerm$covariate,csvFerm$GPUs,csvFerm$streams))
csvFermSort=csvFerm[index,]

index=with(csvDouble,order(csvDouble$individuals,csvDouble$covariate,csvDouble$GPUs,csvDouble$streams))
csvDoubleSort=csvDouble[index,]

ySize=rows
data=array(0,dim=c(ySize))

####Vs Ferm
for(i in 1:rows){
  data[i]=csvNormSort$cueira_calc[i]/csvFermSort$cueira_calc[i]
}
  
xrange=range(1:ySize)
yrange=range(c(0,data))

file=paste("/home/daniel/Project/Results/ferm_calc","", sep="")
file=paste(file,".png", sep="")
png(file)

plot(xrange, yrange, type="n", xlab="", ylab="Less synchronise/always synchronise", cex.axis=s_ax, cex.lab=s_lab, xaxt = "n")
lines(1:ySize,data,type="p",lwd=2)#,lty=lty_a[gpu],pch=pch_a[gpu])
title("Relative Calculation Time",cex.main=s_title)
abline(h=1)

for(i in 0:16){
  abline(v=i*16,col="black")
}

for(i in 0:4){
  abline(v=i*64,col="darkred",lwd=2)
}

text=c("2000","10000", "100000", "200000")
mtext(text,1,line=1,at=c(32,96,160,224))

for(i in 0:3){
  text=c("0","5", "10", "20")
  mtext(text,1,line=0,at=c(8+64*i,24+64*i,40+64*i,56+64*i))
}

dev.off()

####VS Double
for(i in 1:rows){
  data[i]=csvNormSort$cueira_calc[i]/csvDoubleSort$cueira_calc[i]
}

xrange=range(1:ySize)
yrange=range(c(0,data))

file=paste("/home/daniel/Project/Results/double_calc","", sep="")
file=paste(file,".png", sep="")
png(file)

plot(xrange, yrange, type="n", xlab="", ylab="Single precision/double precision", cex.axis=s_ax, cex.lab=s_lab, xaxt = "n")
lines(1:ySize,data,type="p",lwd=2)#,lty=lty_a[gpu],pch=pch_a[gpu])
title("Relative Calculation Time",cex.main=s_title)
abline(h=1)

for(i in 0:16){
  abline(v=i*16,col="black")
}

for(i in 0:4){
  abline(v=i*64,col="darkred",lwd=2)
}

text=c("2000","10000", "100000", "200000")
mtext(text,1,line=1,at=c(32,96,160,224))

for(i in 0:3){
  text=c("0","5", "10", "20")
  mtext(text,1,line=0,at=c(8+64*i,24+64*i,40+64*i,56+64*i))
}

dev.off()