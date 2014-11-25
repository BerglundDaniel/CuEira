#!/usr/bin/python

import sys

sys.stderr.write('Processing results\n')

fname="/home/daniel/Project/Results/run_out.txt"

outFile=open(fname+"_out.csv",'w')

#Write header
outFile.write("snp, individuals, environment, covariate, seed, GPUs, streams")
outFile.write(",FamReaderRead, FanReaderOutcomes, FamReaderTot, ReadEnv, ReadSNPInfo, cueira_init")
outFile.write(", Recode, Next, ReadSNP, ApplyStatModel")
outFile.write(", LR_Tot, LR_GPU, LR_CPU, LR_transferFromDevice, LR_transferToDevice")
outFile.write(", LR_Config_transferToDevice, cueira_calc, ResultWriterLock, QueueLock, cueira_cleanup, cueira_tot")

with open(fname) as f:
    for line in f:
      line=line.rstrip("\n\r")

      if("next_file" in line):
         outFile.write("\n")
         fileBatch=line[10:]

         s_index=fileBatch.find("s")
         s_index_e=fileBatch.find(" ",s_index)
         s=fileBatch[s_index+1:s_index_e]

         i_index=fileBatch.find("i")
         i_index_e=fileBatch.find(" ",i_index)
         i=fileBatch[i_index+1:i_index_e]

         e_index=fileBatch.find("e")
         e_index_e=fileBatch.find(" ",e_index)
         e=fileBatch[e_index+1:e_index_e]

         r_index=fileBatch.find("r")
         r_index_e=fileBatch.find(" ",r_index)
         r=fileBatch[r_index+1:r_index_e]

         c_index=fileBatch.find("c")
         c_index_e=fileBatch.find(" ",c_index)
         c=fileBatch[c_index+1:c_index_e]

         gpu_index=fileBatch.find("gpu")
         gpu_index_e=fileBatch.find(" ",gpu_index)
         gpu=fileBatch[gpu_index+3:gpu_index_e]

         stream_index=fileBatch.find("stream")
         stream=fileBatch[stream_index+6:]

         outFile.write(s+","+i+","+e+","+c+","+r+','+gpu+','+stream)

      #CuEira stuff
      if("FamReader to read file:" in line):
         index=line.find(":")
         end=line.find("second")
         fam_read=line[index+2:end-1]
         outFile.write(","+fam_read)

      if("FamReader to create outcomes:" in line):
         index=line.find(":")
         end=line.find("second")
         fam_out=line[index+2:end-1]
         outFile.write(","+fam_out)

      if("reading personal data:" in line):
         index=line.find(":")
         end=line.find("second")
         fam_tot=line[index+2:end-1]
         outFile.write(","+fam_tot)

      if("reading environment information:" in line):
         index=line.find(":")
         end=line.find("seconds")
         env_read=line[index+2:end-1]
         outFile.write(","+env_read)

      if("reading snp information:" in line):
         index=line.find(":")
         end=line.find("second")
         snp_read=line[index+2:end-1]
         outFile.write(","+snp_read)

      if("initialisation:" in line):
         index=line.find(":")
         end=line.find("second")
         init=line[index+2:end-1]
         outFile.write(","+init)

      if("calculations:" in line):
         index=line.find(":")
         end=line.find("second")
         calc=line[index+2:end-1]
         outFile.write(","+calc)

      if("ResultWriter, time spent waiting at locks:" in line):
         index=line.find(":")
         end=line.find("second")
         res_lock=line[index+2:end-1]
         outFile.write(","+res_lock)

      if("DataQueue, time spent waiting at locks:" in line):
         index=line.find(":")
         end=line.find("second")
         queue_lock=line[index+2:end-1]
         outFile.write(","+queue_lock)

      if("Time for cleanup:" in line):
         index=line.find(":")
         end=line.find("second")
         cleanup=line[index+2:end-1]
         outFile.write(","+cleanup)

      if("Complete, time" in line):
         index=line.find(":")
         end=line.find("second")
         cueira_tot=line[index+2:end-1]
         outFile.write(","+cueira_tot)

      if("Time spent CudaLR" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent GPU" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent CPU" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("LR transferFromDevice" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("LR transferToDevice" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("LRConfig transferToDevice" in line):
         index=line.find(":")
         end=line.find("millisecond")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent recode" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent next" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent read snp" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

      if("Time spent statistic model" in line):
         index=line.find(":")
         end=line.find("second")
         time=line[index+2:end-1]

         outFile.write(","+time)

sys.stderr.write('Done results\n')

