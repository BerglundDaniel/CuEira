#!/usr/bin/python

import sys, os

sys.stderr.write('Starting batch\n')

prefix="/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/"

files=["s100_i100000_e1_r0"]

for f in files:
   sys.stderr.write("next_file "+f+"\n")
   file=prefix+f

   #CuEira
   sys.stderr.write("CuEira\n")
   os.system("/cfs/zorn/nobackup/d/dabergl/CuEira_zorn/build/bin/CuEira -m 0 -b "+file+" -e "+file+"_env.txt -x indid -o "+file+"_out.txt")

   #Geisa
   print("Geisa\n")
   os.system("time -p /cfs/zorn/nobackup/d/dabergl/Geisa/Geisa/geisa_org/dist/geisa -b "+file+" -i "+file+"_env_cov.txt -o "+file+"_geisa_output_dir")

sys.stderr.write('Done batch\n')

