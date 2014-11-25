#!/usr/bin/python

import sys, os

sys.stderr.write('Starting batch\n')

o_dir="/cfs/zorn/nobackup/d/dabergl/Results/"
d_dir="/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/"

s_a=["10000","100000","500000"]
i_a=["10000","100000","200000"]
cov_a=["0","10","20"]
gpu_a=["1","2","3","4"]
streams_a=["1","2","3","4"]

for s in s_a:
   for i in i_a:
      for cov in cov_a:
         for gpu in gpu_a:
            for stream in streams_a:
               sys.stderr.write("next_file "+'s'+str(s)+' i'+str(i)+' e1'+' r0'+' c'+cov+" gpu"+gpu+" stream"+stream+"\n")

               f='s'+str(s)+'_i'+str(i)+'_e1'+'_r0'+'_c0'
               covF='s'+str(s)+'_i'+str(i)+'_e1'+'_r0'+'_c'+cov

               infile=d_dir+"s_"+str(s)+"/"+f
               covFile=d_dir+"s_"+str(s)+"/"+covF
               outfile=o_dir+covF+"_gpu"+gpu+"_stream"+stream+"_out.txt"

               if cov=="0":
                  os.system("/cfs/zorn/nobackup/d/dabergl/CuEira_zorn/build/bin/CuEira -m 0 -b "+infile+" -e "+infile+"_env.txt -x indid -o "+outfile+" --nstreams "+stream+" --ngpus "+gpu)
               else:
                  os.system("/cfs/zorn/nobackup/d/dabergl/CuEira_zorn/build/bin/CuEira -m 0 -b "+infile+" -e "+infile+"_env.txt -x indid -c "+covFile+"_cov.txt -z indid"+" -o "+outfile+" --nstreams "+stream+" --ngpus "+gpu)



sys.stderr.write('Done batch\n')
