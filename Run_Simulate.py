#!/usr/bin/python

import sys

sys.stderr.write('Starting batch')

prefix="/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/"

files=[s100_i100000_e1_r0]

for f in files:
   sys.stderr.write("next "+f+"\n")
   file=prefix+file

   #CuEira
   sys.stderr.write("CuEira\n")
   os.system(/cfs/zorn/nobackup/d/dabergl/CuEira_zorn/build/bin/CuEira -m 0 -b file -e file+"_env.txt" -x indid -o file+"_out.txt")

   #Geisa
   print("Geisa\n")
   os.system(time /cfs/zorn/nobackup/d/dabergl/Geisa/Geisa/geisa_org/dist/geisa -b  file -i file_env_cov.txt -o file_geisa_output_dir)

sys.stderr.write('Done batch')

