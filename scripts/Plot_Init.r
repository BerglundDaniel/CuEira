#csv=read.csv("/home/daniel/Project/Results/10streams/less_sync/10streams_10ks_10ki_prof_noferm.csv", header = TRUE, sep = ",",dec = ".")
csv=read.csv("/home/daniel/Project/Results/10streams/less_sync_back/10streams_10ks_10ki_prof_noferm_backwards.csv", header = TRUE, sep = ",",dec = ".")


rows=nrow(csv)

lty_a=1:4
pch_a=18:21
s_ax=1.2
s_lab=1.3
s_title=1.3
s_legend=1.1

res=array(0,dim=c(rows,2))

for(row in 1:rows){
  res[row,1]=csv$FanReaderOutcomes[row]
  res[row,2]=csv$cueira_init[row]
}

xrange=range(1:rows)
yrange=range(res)

file="/home/daniel/Project/Results/1file_10times.png"
png(file)

plot(xrange, yrange, xlab="", ylab="Seconds", cex.axis=s_ax, cex.lab=s_lab, xaxt = "n", type="n")

lines(1:rows,res[1:rows,2],type="b",lwd=2,lty=lty_a[2],pch=pch_a[1])
lines(1:rows,res[1:rows,1],type="b",lwd=2,lty=lty_a[1],pch=pch_a[2])

title("Initialisation Time",cex.main=s_title)
legend(27.5, yrange[2], c("Total Time","FamReader"), cex=s_legend, pch=pch_a, lty=lty_a)

text=c("4 GPU","3 GPUs", "2 GPUs", "1 GPUs")
axis(1, at=1:40, labels=c(10:1,10:1,10:1,10:1))
mtext(text,1,line=3,at=c(5,15,25,35))
for(i in 0:2){
  abline(v=10.5+10*i,col="black")
}

mtext("Number of",1,line=0.5,at=-3)
mtext("streams",1,line=1.5,at=-3)

dev.off()