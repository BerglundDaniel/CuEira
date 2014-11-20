#!/usr/bin/python

import sys, os

sys.stderr.write('Starting batch\n')

o_dir="/cfs/zorn/nobackup/d/dabergl/Results/"

#prefix="/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_100/"
#files=["s100_i2000_e1_r0", "s100_i10000_e1_r0", "s100_i50000_e1_r0", "s100_i100000_e1_r0", "s100_i200000_e1_r0"]

#prefix="/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_10000/"
#files=["s10000_i2000_e1_r0", "s10000_i10000_e1_r0", "s10000_i50000_e1_r0", "s10000_i100000_e1_r0", "s10000_i200000_e1_r0"]

prefix=""
files=["/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_100/s100_i10000_e1_r0", "/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_10000/s10000_i10000_e1_r0", "/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_100000/s100000_i10000_e1_r0", "/cfs/zorn/nobackup/d/dabergl/Data/Simulated_Data/s_500000/s500000_i10000_e1_r0"]

for f in files:
   sys.stderr.write("next_file "+f+"\n")
   file=prefix+f

   #CuEira
   sys.stderr.write("CuEira\n")
   os.system("/cfs/zorn/nobackup/d/dabergl/CuEira_zorn/build/bin/CuEira -m 0 -b "+file+" -e "+file+"_env.txt -x indid -o "+o_dir+f+"_out.txt")

   #Geisa
   #sys.stderr.write("Geisa\n")
   #os.system("time -p /cfs/zorn/nobackup/d/dabergl/Geisa/Geisa/geisa_org/dist/geisa -b "+file+" -i "+file+"_env_cov.txt -o "+o_dir+f+"_geisa_output_dir")

sys.stderr.write('Done batch\n')

