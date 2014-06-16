#include "BedReader.h"

namespace CuEira {
namespace FileIO {

BedReader::BedReader(const Configuration& configuration, const PersonHandler& personHandler, const int numberOfSNPs) :
    configuration(configuration), personHandler(personHandler), bedFileStr(configuration.getBedFilePath()), geneticModel(
        configuration.getGeneticModel()), numberOfSNPs(numberOfSNPs), numberOfIndividualsToInclude(
        personHandler.getNumberOfIndividualsToInclude()), numberOfIndividualsTotal(
        personHandler.getNumberOfIndividualsTotal()) {

  std::ifstream bedFile;
  openBedFile(bedFile);

  //Read header to check version and mode.
  char buffer[headerSize];
  bedFile.seekg(0, std::ios::beg);
  bedFile.read(buffer, headerSize);

  if(!bedFile){
    std::ostringstream os;
    os << "Problem reading header of bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  //Check version
  if(!(buffer[0] == 108 && buffer[1] == 27)){ //If first byte is 01101100 and second is 00011011 then we have a bed file
    std::ostringstream os;
    os << "Provided bed file is not a bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  //Check mode
  if(buffer[2] == 1){      // 00000001
#ifdef DEBUG
      std::cerr << "Bed file is SNP major" << std::endl;
#endif
    mode = SNPMAJOR;
  }else if(buffer[2] == 0){      // 00000000
#ifdef DEBUG
      std::cerr << "Bed file is individual major" << std::endl;
#endif
    mode = INDIVIDUALMAJOR;
  }else{
    std::ostringstream os;
    os << "Unknown major mode of the bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  closeBedFile(bedFile);
}

BedReader::BedReader(const Configuration& configuration, const PersonHandler& personHandler) :
    geneticModel(DOMINANT), numberOfSNPs(0), numberOfIndividualsTotal(0), numberOfIndividualsToInclude(0), configuration(
        configuration), personHandler(personHandler) {

}

BedReader::~BedReader() {

}

Container::SNPVector* BedReader::readSNP(SNP& snp) const {
  std::ifstream bedFile;
  int numberOfAlleleOneCase = 0;
  int numberOfAlleleTwoCase = 0;
  int numberOfAlleleOneControl = 0;
  int numberOfAlleleTwoControl = 0;
  int numberOfAlleleOneAll = 0;
  int numberOfAlleleTwoAll = 0;
  const int snpPos = snp.getPosition();

  //Initialise vector
  std::vector<int>* snpDataOriginal = new std::vector<int>(numberOfIndividualsToInclude);

  openBedFile(bedFile);

  //Read depending on the mode
  if(mode == SNPMAJOR){
    const int numberOfBitsPerRow = numberOfIndividualsTotal * 2;
    //Each individuals genotype is stored as 2 bits, there are no incomplete bytes so we have to round the number up
    const int numberOfBytesPerRow = std::ceil(((double) numberOfBitsPerRow) / 8);
    int readBufferSize; //Number of bytes to read per read

    if(numberOfBytesPerRow < readBufferSizeMaxSNPMAJOR){
      readBufferSize = numberOfBytesPerRow;
    }else{
      readBufferSize = readBufferSizeMaxSNPMAJOR;
    }

    //Number of reads we have to do to read all the info for the SNP
    const int numberOfReads = std::ceil(((double) numberOfBytesPerRow) / readBufferSize);

    //The number of bits at the start(due to the reversing of the bytes) of the last byte that we don't care about
    const int numberOfUninterestingBitsAtEnd = (8 * numberOfBytesPerRow) - numberOfBitsPerRow;

    //Read the file until we read all the info for this SNP
    for(int readNumber = 1; readNumber <= numberOfReads; ++readNumber){

      //We have to fix the buffersize if it's the lastread and if we couldn't read the whole row at once
      if(readNumber == numberOfReads && readNumber != 1){
        readBufferSize = numberOfBytesPerRow - readBufferSize * (numberOfReads - 1);
      }

      char buffer[readBufferSize];
      long int seekPos = headerSize + numberOfBytesPerRow * snpPos + readBufferSize * (readNumber - 1);

      bedFile.seekg(seekPos);
      bedFile.read(buffer, readBufferSize);
      if(!bedFile){
        std::ostringstream os;
        os << "Problem reading SNP " << snp.getId().getString() << " from bed file " << bedFileStr << std::endl;
        const std::string& tmp = os.str();
        throw FileReaderException(tmp.c_str());
      }

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
        for(int bitPairNumber = 0; bitPairNumber < numberOfBitPairsPerByte; ++bitPairNumber){
          //It's in reverse due to plinks format that has the bits in each byte in reverse
          int posInByte = 2 * bitPairNumber;
          bool firstBit = getBit(currentByte, posInByte);
          bool secondBit = getBit(currentByte, posInByte + 1);

          //Which person does this information belong to?
          int personRowFileNumber = (readNumber - 1) * readBufferSize + byteNumber * 4 + bitPairNumber;

          const Person& person = personHandler.getPersonFromRowAll(personRowFileNumber);

          if(person.getInclude()){ //If the person shouldn't be included we will skip it
            Phenotype phenotype = person.getPhenotype();
            int currentPersonRow = personHandler.getRowIncludeFromPerson(person);

            //If we are missing the genotype for at least one individual(that should be included) we excluded the SNP
            if(firstBit && !secondBit){
#ifdef DEBUG
              std::cerr << "Excluding SNP " << snp.getId().getString() << std::endl;
#endif
              snp.setInclude(false);
              closeBedFile(bedFile);
              return nullptr; //Since we are going to exclude this SNP there is no point in reading more data.
            }/* if check missing */

            //Store the genotype as 0,1,2 until we can recode it. We have to know the risk allele before we can recode.
            //Also increase the counters for the alleles if it is a case.
            if(!firstBit && !secondBit){
              //Homozygote primary
              (*snpDataOriginal)[currentPersonRow] = 0;
              numberOfAlleleOneAll += 2;

              if(phenotype == AFFECTED){
                numberOfAlleleOneCase += 2;
              }else{
                numberOfAlleleOneControl += 2;
              }
            }else if(!firstBit && secondBit){
              //Hetrozygote
              (*snpDataOriginal)[currentPersonRow] = 1;
              numberOfAlleleOneAll++;
              numberOfAlleleTwoAll++;

              if(phenotype == AFFECTED){
                numberOfAlleleOneCase++;
                numberOfAlleleTwoCase++;
              }else{
                numberOfAlleleOneControl++;
                numberOfAlleleTwoControl++;
              }
            }else if(firstBit && secondBit){
              //Homozygote secondary
              (*snpDataOriginal)[currentPersonRow] = 2;
              numberOfAlleleTwoAll += 2;

              if(phenotype == AFFECTED){
                numberOfAlleleTwoCase += 2;
              }else{
                numberOfAlleleTwoControl += 2;
              }
            }

          }/* if person include */

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

  closeBedFile(bedFile);

  //Convert numbers to frequencies
  const int numberOfAllelesInPopulation = (numberOfIndividualsToInclude * 2);
  const int numberOfAllelesInCase = numberOfAlleleOneCase + numberOfAlleleTwoCase;
  const int numberOfAllelesInControl = numberOfAlleleOneControl + numberOfAlleleTwoControl;

  const double alleleOneCaseFrequency = (double) numberOfAlleleOneCase / numberOfAllelesInCase;
  const double alleleTwoCaseFrequency = (double) numberOfAlleleTwoCase / numberOfAllelesInCase;
  const double alleleOneControlFrequency = (double) numberOfAlleleOneControl / numberOfAllelesInControl;
  const double alleleTwoControlFrequency = (double) numberOfAlleleTwoControl / numberOfAllelesInControl;
  const double alleleOneAllFrequency = (double) numberOfAlleleOneAll / numberOfAllelesInPopulation;
  const double alleleTwoAllFrequency = (double) numberOfAlleleTwoAll / numberOfAllelesInPopulation;

  snp.setCaseAlleleFrequencies(alleleOneCaseFrequency, alleleTwoCaseFrequency);
  snp.setControlAlleleFrequencies(alleleOneControlFrequency, alleleTwoControlFrequency);
  snp.setAllAlleleFrequencies(alleleOneAllFrequency, alleleTwoAllFrequency);

  //Check which allele is most frequent in cases
  if(alleleOneCaseFrequency == alleleTwoCaseFrequency){
#ifdef DEBUG
    std::cerr << "SNP " << snp.getId().getString() << " has equal case allele frequency." << std::endl;
#endif
    if(alleleOneControlFrequency == alleleTwoControlFrequency){
      std::cerr << "SNP " << snp.getId().getString()
          << " has equal control and case allele frequency, setting allele one as risk." << std::endl;
      snp.setRiskAllele(ALLELE_ONE);
    }else if(alleleOneControlFrequency < alleleTwoControlFrequency){
      snp.setRiskAllele(ALLELE_ONE);
    }else{
      snp.setRiskAllele(ALLELE_TWO);
    }
  }else if(alleleOneCaseFrequency > alleleTwoCaseFrequency){
    if(alleleOneCaseFrequency >= alleleOneControlFrequency){
      snp.setRiskAllele(ALLELE_ONE);
    }else{
      snp.setRiskAllele(ALLELE_TWO);
    }
  }else{
    if(alleleTwoCaseFrequency >= alleleTwoControlFrequency){
      snp.setRiskAllele(ALLELE_TWO);
    }else{
      snp.setRiskAllele(ALLELE_ONE);
    }
  }

  //Calculate MAF
  double minorAlleleFrequency;

  if(alleleOneAllFrequency == alleleTwoAllFrequency){
    minorAlleleFrequency = alleleOneAllFrequency;
  }else if(alleleOneAllFrequency > alleleTwoAllFrequency){
    minorAlleleFrequency = alleleTwoAllFrequency;
  }else{
    minorAlleleFrequency = alleleOneAllFrequency;
  }
  snp.setMinorAlleleFrequency(minorAlleleFrequency);

  return new Container::SNPVector(snpDataOriginal, snp, geneticModel);
} /* readSNP */

// position in range 0-7
bool BedReader::getBit(unsigned char byte, int position) const {
  return (byte >> position) & 0x1; //Shift the byte to the right so we have bit at the position as the last bit and then use bitwise and with 00000001
}

void BedReader::openBedFile(std::ifstream& bedFile) const {
  bedFile.open(bedFileStr, std::ifstream::binary);
  if(!bedFile){
    std::ostringstream os;
    os << "Problem opening bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

void BedReader::closeBedFile(std::ifstream& bedFile) const {
  if(bedFile.is_open()){
    bedFile.close();
  }
  if(!bedFile){
    std::ostringstream os;
    os << "Problem closing bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
