#!/usr/bin/python

import sys, getopt, random

message='Simulate_Data.py -o <outputfiles> -s <number of snps> -i <number of individuals> -e <number of environment> -c <number of covariates> -r <seed>'

argv=sys.argv[1:]

if len(argv)==0:
   print message
   sys.exit(2)

numberOfSNPs=0
numberOfIndividuals=0
numberOfEnvironment=1
numberOfCovariates=0
seed=0

try:
   opts, args = getopt.getopt(argv,"hs:i:e:c:r:",[])
except getopt.GetoptError:
   print message
   sys.exit(2)

for opt, arg in opts:
   if opt == '-h':
      print message
      sys.exit()
   elif opt == ("-s"):
      numberOfSNPs = int(float(arg))
   elif opt == ("-i"):
      numberOfIndividuals = int(arg)
   elif opt == ("-e"):
      numberOfEnvironment = int(arg)
   elif opt == ("-c"):
      numberOfCovariates = int(arg)
   elif opt == ("-r"):
      seed = int(arg)

print 'Starting'

outputfile = 's'+str(numberOfSNPs)+'_i'+str(numberOfIndividuals)+'_e'+str(numberOfEnvironment)+'_r'+str(seed)+'_c'+str(numberOfCovariates)

random.seed(seed)
stateRandom=random.getstate()

#fam
random.setstate(stateRandom)

famFile=open(outputfile+'.fam','w')
for i in range(0, numberOfIndividuals):
   #fam id 0 0 sex status
   s=d=random.randint(1,2)
   status=-1
   if i<numberOfIndividuals/2:
      status=2
   else:
      status=1
   famFile.write('fam'+str(i)+' per'+str(i)+' 0 0 '+str(s)+' '+str(status)+'\n')

famFile.close()

#bim
random.setstate(stateRandom)
bimFile=open(outputfile+'.bim','w')
for i in range(0, numberOfSNPs):
   #chrom id 0 i A G
   bimFile.write('1 snp'+str(i)+' 0 '+str(i)+' A G\n')

bimFile.close()

#bed
random.setstate(stateRandom)
bedFile=open(outputfile+'.bed','wb')

#header
bedFile.write(bytearray([108,27,1]))

for i in range(0, numberOfSNPs):
  for j in range(0, numberOfIndividuals, 4):  
      s=4
      b=0
      if j>numberOfIndividuals-4:
         s=numberOfIndividuals-j
      for k in range(0,s):
         r=random.randint(0,2)
         if r>0:
            r+=1
         b+=r*4**k
      byte=bytearray([b])
      bedFile.write(byte)

bedFile.close()

#env cov
random.setstate(stateRandom)

envCovFile=open(outputfile+'_env_cov.txt','w')
envFile=open(outputfile+'_env.txt','w')
if numberOfCovariates!=0:
   covFile=open(outputfile+'_cov.txt','w')

#headers
envCovFile.write('indid')
envFile.write('indid')
if numberOfCovariates!=0:
   covFile.write('indid')

for j in range(0, numberOfEnvironment):
   envFile.write('\tenv')#+str(j))
   envCovFile.write('\tenv')#+str(j))

for j in range(0, numberOfCovariates):
   if numberOfCovariates!=0:
      covFile.write('\tcov'+str(j))
   envCovFile.write('\tcov'+str(j))

envCovFile.write('\n')
envFile.write('\n')
if numberOfCovariates!=0:
   covFile.write('\n')

#lines
for i in range(0, numberOfIndividuals):
   envCovFile.write('per'+str(i))
   envFile.write('per'+str(i))
   if numberOfCovariates!=0:
      covFile.write('per'+str(i))

   for j in range(0, numberOfEnvironment):
      d=random.randint(0,1)
      envCovFile.write('\t'+str(d))
      envFile.write('\t'+str(d))
   for j in range(0, numberOfCovariates):
      d=random.randint(0,1)
      envCovFile.write('\t'+str(d))
      if numberOfCovariates!=0:
         covFile.write('\t'+str(d))

   envCovFile.write('\n')
   envFile.write('\n')
   if numberOfCovariates!=0:
      covFile.write('\n')

envCovFile.close()
envFile.close()
if numberOfCovariates!=0:
   covFile.close()

print 'Done'

