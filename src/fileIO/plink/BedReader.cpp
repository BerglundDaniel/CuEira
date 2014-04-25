#include "BedReader.h"

namespace CuEira {
namespace FileIO {

BedReader::BedReader(const Configuration& configuration, const Container::HostVector& outcomes,
    const int numberOfIndividuals, const int numberOfSNPs) :
    configuration(configuration), outcomes(outcomes), bedFileStr(configuration.getBedFilePath()), geneticModel(
        configuration.getGeneticModel()), numberOfIndividuals(numberOfIndividuals), numberOfSNPs(numberOfSNPs) {

  openBedFile();

  //Read header to check version and mode.
  try{
    char buffer[headerSize];
    bedFile.seekg(0, std::ios::beg);
    bedFile.read(buffer, headerSize);

    //Check version
    if(!(buffer[0] == 108 && buffer[1] == 27)){ //If first byte is 01101100 and second is 00011011 then we have a bed file
      std::ostringstream os;
      os << "Provided bed file is not a bed file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

    //Check mode
    if(buffer[2] == 1){      // 00000001
      std::cerr << "Bed file is SNP major" << std::endl;
      mode = SNPMAJOR;
    }else if(buffer[2] == 0){      // 00000000
      std::cerr << "Bed file is individual major" << std::endl;
      mode = INDIVIDUALMAJOR;
    }else{
      std::ostringstream os;
      os << "Unknown major mode of the bed file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading header of bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  closeBedFile();

}

BedReader::~BedReader() {

}

Container::HostVector BedReader::readSNP(SNP& snp) const {
  //Initialise vector
  Container::HostVector SNPVector(numberOfIndividuals);

  openBedFile();
  size_t alleleOneCaseFrequency = 0;
  size_t alleleTwoCaseFrequency = 0;
  size_t alleleOneFrequency = 0;
  size_t alleleTwoFrequency = 0;

  //Read SNP
  try{
    //Read depending on the mode
    if(mode == SNPMAJOR){
      const int numberOfBitsPerRow = numberOfIndividuals * 2;
      //Each individuals genotype is stored as 2 bits, there are no incomplete bytes so we have to round the number up
      const int numberOfBytesPerRow = std::ceil(numberOfBitsPerRow / 8);
      int readBufferSize; //Number of bytes to read per read
      const int numberOfUninterestingBitsAtEnd = (8 * numberOfBytesPerRow) - numberOfBitsPerRow; //The number of bits at the start(due to the reversing of the bytes) of the last byte that we don't care about

      if(numberOfBytesPerRow < readBufferSizeMaxSNPMAJOR){
        readBufferSize = numberOfBytesPerRow;
      }else{
        readBufferSize = readBufferSizeMaxSNPMAJOR;
      }

      const int numberOfReads = std::ceil(numberOfBytesPerRow / readBufferSize); //Number of reads we have to do to read all the info for the SNP

      //Read the file until we read all the info for this SNP
      for(int readNumber = 1; readNumber <= numberOfReads; ++readNumber){
        char buffer[readBufferSize];
        bedFile.seekg(headerSize + numberOfBytesPerRow * readNumber);
        bedFile.read(buffer, readBufferSize);

        //Go through all the bytes in this read
        for(int byteNumber = 0; byteNumber < readBufferSize; ++byteNumber){
          char currentByte = buffer[byteNumber];

          int numberOfBitPairsPerByte;
          if(readNumber == numberOfReads && byteNumber == (readBufferSize - 1)){ //Are we at the last byte in the last read?
            numberOfBitPairsPerByte = (8 - numberOfUninterestingBitsAtEnd) / 2;
          }else{
            numberOfBitPairsPerByte = 4;
          }

          //Go through all the pairs of bit in the byte
          for(int bitPairNumber = 1; bitPairNumber <= numberOfBitPairsPerByte; ++bitPairNumber){
            //It's in reverse due to plinks format that has the bits in each byte in reverse
            int posInByte = 8 - 2 * bitPairNumber;
            bool firstBit = getBit(currentByte, posInByte + 1);
            bool secondBit = getBit(currentByte, posInByte);
            //The position in the vector where we are going to store the geneotype for this individual
            int SNPPos = (readNumber - 1) * readBufferSize + byteNumber * 4 + bitPairNumber - 1;
            int genotype;

            if(firstBit && !secondBit){ //The genotype is missing for this individual
              //The genotype is missing for this individual so we will exclude the SNP it from the analysis.
              excludeSNP(snp);
              closeBedFile();
              return SNPVector; //Since we are going to exclude this SNP there is no point in reading more data.
            }/* if check missing */

            //Store the genotype as 0,1,2 until we can recode it. We have to know the risk allel before we can recode to 0 and 1.
            //Also increase the counters for the alleles if it is a case.
            if(!firstBit && !secondBit){
              //Homozygote primary
              SNPVector[SNPPos] = 0;
              alleleOneFrequency += 2;

              if(outcomes[SNPPos] == 1){
                alleleOneCaseFrequency += 2;
              }
            }else if(!firstBit && secondBit){
              //Hetrozygote
              SNPVector[SNPPos] = 1;
              alleleOneFrequency++;
              alleleTwoFrequency++;

              if(outcomes[SNPPos] == 1){
                alleleOneCaseFrequency++;
                alleleTwoCaseFrequency++;
              }
            }else if(firstBit && secondBit){
              //Homozygote secondary
              SNPVector[SNPPos] = 2;
              alleleTwoFrequency += 2;

              if(outcomes[SNPPos] == 1){
                alleleTwoCaseFrequency += 2;
              }
            }

            SNPVector[SNPPos] = genotype;
          }/* for bitPairNumber */

        }/* for byteNumber */

      }/* for readNumber */

    }else if(mode == INDIVIDUALMAJOR){
      //TODO
      throw FileReaderException("Individual major mode is not implemented yet.");
    }else{
      std::ostringstream os;
      os << "No mode set for the file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem reading SNP " << snp.getId().getString() << " from bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  closeBedFile();

  //Check which allele is most frequent in cases
  if(alleleOneCaseFrequency == alleleTwoCaseFrequency){
    snp.setRiskAllele(ALLELE_ONE);
    std::cerr << "WARNING: SNP " << snp.getId().getString()
        << " has equal case allele frequency. Setting allele one as risk allele." << std::endl;
  }else if(alleleOneCaseFrequency > alleleTwoCaseFrequency){
    snp.setRiskAllele(ALLELE_ONE);
  }else{
    snp.setRiskAllele(ALLELE_TWO);
  }

  //Calculate MAF
  double minorAlleleFrequencyThreshold = configuration.getMinorAlleleFrequencyThreshold();
  if(minorAlleleFrequencyThreshold != 0){ //No point in calculating it if the threshold is zero since all SNPS will pass that.
    size_t numberOfMinorAllele;
    if(snp.getRiskAllele() == ALLELE_ONE){
      numberOfMinorAllele = alleleOneFrequency;
    }else if(snp.getRiskAllele() == ALLELE_TWO){
      numberOfMinorAllele = alleleTwoFrequency;
    }else{
      throw FileReaderException("Unknown risk allele. This should not happen.");
    }

    double minorAlleleFrequency = numberOfIndividuals * 2 / numberOfMinorAllele;
    snp.setMinorAlleleFrequency(minorAlleleFrequency);
    if(minorAlleleFrequencyThreshold > minorAlleleFrequency){
      excludeSNP(snp);
    }
  }/* if maf treshold!=0 */

  //Recode based on which allele is the risk
  for(int i = 0; i < numberOfIndividuals; ++i){
    if(geneticModel == DOMINANT){
      if(snp.getRiskAllele() == ALLELE_ONE){
        //TODO
      }else if(snp.getRiskAllele() == ALLELE_TWO){
        //TODO
      }else{
        throw FileReaderException("Unknown risk allele. This should not happen.");
      }
    }else if(geneticModel == RECESSIVE){
      if(snp.getRiskAllele() == ALLELE_ONE){
        //TODO
      }else if(snp.getRiskAllele() == ALLELE_TWO){
        //TODO
      }else{
        throw FileReaderException("Unknown risk allele. This should not happen.");
      }
    }else{
      std::ostringstream os;
      os << "Unknown genetic model " << geneticModel << std::endl;
      const std::string& tmp = os.str();
      throw std::invalid_argument(tmp.c_str());
    }
  }/* for i[0,numberOfIndividuals] */

  return SNPVector;

}/* readSNP */

// position in range 0-7
bool BedReader::getBit(unsigned char byte, int position) const {
  return (byte >> position) & 0x1; //Shift the byte to the right so we have bit at the position as the last bit and then use bitwise and with 00000001
}

void BedReader::excludeSNP(SNP& snp) const {
  snp.setInclude(false);
  return;
}

void BedReader::openBedFile() {
  try{
    bedFile.open(bedFileStr, std::ifstream::binary);
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem opening bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

void BedReader::closeBedFile() {
  try{
    if(!bedFile.is_open()){

      bedFile.close();
    }
  } catch(const std::ios_base::failure& exception){
    std::ostringstream os;
    os << "Problem closing bed file " << bedFileStr << std::endl;
#ifdef DEBUG
    os << exception.what();
#endif
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  return;
}

} /* namespace FileIO */
} /* namespace CuEira */
