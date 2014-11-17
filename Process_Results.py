#!/usr/bin/python

import sys

sys.stderr.write('Processing results\n')

fname="/home/daniel/Project/Results/out.txt"
numThreads=12
infoPerThread=9

outFile=open(fname+"_out.csv",'w')

threadsInfo=[None]*numThreads*infoPerThread
threadToNum={}

#Write header
outFile.write("snp, individuals, environment, covariate, seed")
outFile.write(",FamReaderRead, FanReaderOutcomes, FamReaderTot, ReadEnv, ReadSNPInfo, cueira_init, cueira_calc, ResultWriterLock, QueueLock, cueira_cleanup, cueira_tot")

outFile.write(", geisa_tot")

for i in range(0,numThreads):
   outFile.write(", LR_Tot, LR_GPU, LR_CPU")
   outFile.write(", Recode, Next, ReadSNP, ApplyStatModel")
   outFile.write(", Thread_TotalTime, Thread_CalcTime")

threadId=0
first=1
with open(fname) as f:
    for line in f:
      line=line.rstrip("\n\r")

      #Check if a new thread
      if("Thread:" in line and threadId<numThreads):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         if(key not in threadToNum):
            threadToNum[key]=threadId
            threadId=threadId+1

      if("next_file" in line):
         if(not first==1):
            for info in threadsInfo:
               outFile.write(","+info)

         outFile.write("\n")
         fileBatch=line[10:]

         s_index=fileBatch.find("s")
         s_index_e=fileBatch.find("_",s_index)
         s=fileBatch[s_index+1:s_index_e]

         i_index=fileBatch.find("i")
         i_index_e=fileBatch.find("_",i_index)
         i=fileBatch[i_index+1:i_index_e]

         e_index=fileBatch.find("e")
         e_index_e=fileBatch.find("_",e_index)
         e=fileBatch[e_index+1:e_index_e]

         r_index=fileBatch.find("r")
         r_index_e=fileBatch.find("_",r_index)
         r=fileBatch[r_index+1:]#r_index_e]

         #c_index=fileBatch.find("c")
         #c=fileBatch[c_index+1:]
         c="0"
         outFile.write(s+","+i+","+e+","+c+","+r)

         first=0

      #CuEira stuff
      if("FamReader to read file:" in line):
         index=line.find(":")
         end=line.find("seconds")
         fam_read=line[index+2:end-1]
         outFile.write(","+fam_read)

      if("FamReader to create outcomes:" in line):
         index=line.find(":")
         end=line.find("seconds")
         fam_out=line[index+2:end-1]
         outFile.write(","+fam_out)

      if("reading personal data:" in line):
         index=line.find(":")
         end=line.find("seconds")
         fam_tot=line[index+2:end-1]
         outFile.write(","+fam_tot)

      if("reading environment information:" in line):
         index=line.find(":")
         end=line.find("seconds")
         env_read=line[index+2:end-1]
         outFile.write(","+env_read)

      if("reading snp information:" in line):
         index=line.find(":")
         end=line.find("seconds")
         snp_read=line[index+2:end-1]
         outFile.write(","+snp_read)

      if("initialisation:" in line):
         index=line.find(":")
         end=line.find("seconds")
         init=line[index+2:end-1]
         outFile.write(","+init)

      if("calculations:" in line):
         index=line.find(":")
         end=line.find("seconds")
         calc=line[index+2:end-1]
         outFile.write(","+calc)

      if("ResultWriter, time spent waiting at locks:" in line):
         index=line.find(":")
         end=line.find("microseconds")
         res_lock=line[index+2:end-1]
         outFile.write(","+res_lock)

      if("DataQueue, time spent waiting at locks:" in line):
         index=line.find(":")
         end=line.find("microseconds")
         queue_lock=line[index+2:end-1]
         outFile.write(","+queue_lock)

      if("Time for cleanup:" in line):
         index=line.find(":")
         end=line.find("seconds")
         cleanup=line[index+2:end-1]
         outFile.write(","+cleanup)

      if("Complete, time" in line):
         index=line.find(":")
         end=line.find("seconds")
         cueira_tot=line[index+2:end-1]
         outFile.write(","+cueira_tot)

      #CuEira threads
      if("Time spent CudaLR" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+0]=time

      if("Time spent GPU" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+1]=time

      if("Time spent CPU" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+2]=time

      if("Time spent recode" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+3]=time

      if("Time spent next" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+4]=time

      if("Time spent read snp" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+5]=time

      if("Time spent statistic model" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("milliseconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+6]=time

      if("TotalTime" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("seconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+7]=time

      if("CalcTime" in line):
         index=line.find(":")
         end=line.find(" ",index+2)
         key=line[index+2:end]

         index=line.find(":", end)
         end=line.find("seconds")
         time=line[index+2:end-1]

         threadsInfo[threadToNum[key]*infoPerThread+8]=time

      #Geisa stuff
      if("real " in line):
         index=line.find(" ")
         geisa_tot=line[index+1:]
         outFile.write(","+geisa_tot)

      #if():
         #g_calc="0"
         #outFile.write(","+g_calc)

      #if():
         #g_tot="0"
         #outFile.write(","+g_tot)

for info in threadsInfo:
   outFile.write(","+info)

sys.stderr.write('Done results\n')

