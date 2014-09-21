#include "BedReader.h"

namespace CuEira {
namespace FileIO {

BedReader::BedReader(const Configuration& configuration, const Container::SNPVectorFactory& snpVectorFactory,
    const AlleleStatisticsFactory& alleleStatisticsFactory, const PersonHandler& personHandler, const int numberOfSNPs) :
    configuration(configuration), snpVectorFactory(snpVectorFactory), alleleStatisticsFactory(alleleStatisticsFactory), personHandler(
        personHandler), bedFileStr(configuration.getBedFilePath()), numberOfSNPs(numberOfSNPs), numberOfIndividualsToInclude(
        personHandler.getNumberOfIndividualsToInclude()), numberOfIndividualsTotal(
        personHandler.getNumberOfIndividualsTotal()), minorAlleleFrequencyThreshold(
        configuration.getMinorAlleleFrequencyThreshold()) {

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

  if(mode == SNPMAJOR){
    numberOfBitsPerRow = numberOfIndividualsTotal * 2;
    //Each individuals genotype is stored as 2 bits, there are no incomplete bytes so we have to round the number up
    numberOfBytesPerRow = std::ceil(((double) numberOfBitsPerRow) / 8);

    //The number of bits at the start of the last byte(due to the reversing of the bytes) that we don't care about
    numberOfUninterestingBitsAtEnd = (8 * numberOfBytesPerRow) - numberOfBitsPerRow;
  }else{
    throw FileReaderException("Individual major mode is not implemented yet.");
  }
}

BedReader::BedReader(const Configuration& configuration, const Container::SNPVectorFactory& snpVectorFactory,
    const AlleleStatisticsFactory& alleleStatisticsFactory, const PersonHandler& personHandler) :
    numberOfSNPs(0), numberOfIndividualsTotal(0), numberOfIndividualsToInclude(0), configuration(configuration), personHandler(
        personHandler), snpVectorFactory(snpVectorFactory), alleleStatisticsFactory(alleleStatisticsFactory), minorAlleleFrequencyThreshold(
        0), numberOfBitsPerRow(0), numberOfBytesPerRow(0), numberOfUninterestingBitsAtEnd(0) {

}

BedReader::~BedReader() {

}

std::pair<const AlleleStatistics*, Container::SNPVector*>* BedReader::readSNP(SNP& snp) {
  std::ifstream bedFile;
  int numberOfAlleleOneCase = 0;
  int numberOfAlleleTwoCase = 0;
  int numberOfAlleleOneControl = 0;
  int numberOfAlleleTwoControl = 0;
  int numberOfAlleleOneAll = 0;
  int numberOfAlleleTwoAll = 0;
  const int snpPos = snp.getPosition();
  bool missingData = false;

  //Initialise vector
  std::vector<int>* snpDataOriginal = new std::vector<int>(numberOfIndividualsToInclude);

  //Read depending on the mode
  if(mode == SNPMAJOR){
    int readBufferSize; //Number of bytes to read per read //TODO REMOVE to numberOfBytesPerRow

    char buffer[numberOfBytesPerRow];
    long int seekPos = headerSize + numberOfBytesPerRow * snpPos;

    openBedFile(bedFile);
    bedFile.seekg(seekPos);
    bedFile.read(buffer, numberOfBytesPerRow);
    if(!bedFile){
      std::ostringstream os;
      os << "Problem reading SNP " << snp.getId().getString() << " from bed file " << bedFileStr << std::endl;
      const std::string& tmp = os.str();
      throw FileReaderException(tmp.c_str());
    }

    closeBedFile(bedFile);

    //Go through all the bytes in this read
    for(int byteNumber = 0; byteNumber < numberOfBytesPerRow; ++byteNumber){
      char currentByte = buffer[byteNumber];

      int numberOfBitPairsPerByte;
      if(byteNumber == (numberOfBytesPerRow - 1)){ //Are we at the last byte
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
        int personRowFileNumber = byteNumber * 4 + bitPairNumber;

        const Person& person = personHandler.getPersonFromRowAll(personRowFileNumber);

        if(person.getInclude()){ //If the person shouldn't be included we will skip it
          Phenotype phenotype = person.getPhenotype();
          int currentPersonRow = personHandler.getRowIncludeFromPerson(person);

          //If we are missing the genotype for at least one individual(that should be included) we excluded the SNP
          if(firstBit && !secondBit){
            missingData = true;
            (*snpDataOriginal)[currentPersonRow] = -1;
          }else{
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
          }/* if check missing */
        }/* if person include */

      }/* for bitPairNumber */

    }/* for byteNumber */

  }else if(mode == INDIVIDUALMAJOR){
    //TODO
    delete snpDataOriginal;
    throw FileReaderException("Individual major mode is not implemented yet.");
  }else{
    std::ostringstream os;
    os << "No mode set for the file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }

  std::vector<int>* numberOfAlleles = new std::vector<int>(6);

  (*numberOfAlleles)[ALLELE_ONE_CASE_POSITION] = numberOfAlleleOneCase;
  (*numberOfAlleles)[ALLELE_TWO_CASE_POSITION] = numberOfAlleleTwoCase;
  (*numberOfAlleles)[ALLELE_ONE_CONTROL_POSITION] = numberOfAlleleOneControl;
  (*numberOfAlleles)[ALLELE_TWO_CONTROL_POSITION] = numberOfAlleleTwoControl;
  (*numberOfAlleles)[ALLELE_ONE_ALL_POSITION] = numberOfAlleleOneAll;
  (*numberOfAlleles)[ALLELE_TWO_ALL_POSITION] = numberOfAlleleTwoAll;

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = new std::pair<const AlleleStatistics*,
      Container::SNPVector*>();
  pair->first = alleleStatisticsFactory.constructAlleleStatistics(numberOfAlleles);

  if(missingData){
    snp.setInclude(MISSING_DATA);
  }

  setSNPInclude(snp, *(pair->first));
  setSNPRiskAllele(snp, *(pair->first));

  if(snp.shouldInclude()){
    pair->second = snpVectorFactory.constructSNPVector(snp, snpDataOriginal);
  }else{
    pair->second = nullptr;
  }

  return pair;
} /* readSNP */

// position in range 0-7
bool BedReader::getBit(unsigned char byte, int position) const {
  return (byte >> position) & 0x1; //Shift the byte to the right so we have bit at the position as the last bit and then use bitwise and with 00000001
}

void BedReader::openBedFile(std::ifstream& bedFile) {
  bedFile.open(bedFileStr, std::ifstream::binary);
  if(!bedFile){
    std::ostringstream os;
    os << "Problem opening bed file " << bedFileStr << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
}

void BedReader::closeBedFile(std::ifstream& bedFile) {
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

void BedReader::setSNPRiskAllele(SNP& snp, const AlleleStatistics& alleleStatistics) const {
  const std::vector<double>& alleleFrequencies = alleleStatistics.getAlleleFrequencies();

  //Check which allele is most frequent in cases
  RiskAllele riskAllele;
  /*
   if((alleleFrequencies[ALLELE_ONE_CASE_POSITION] - alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]) > 0){
   riskAllele = ALLELE_ONE;
   }else{
   riskAllele = ALLELE_TWO;
   }
   */
  /*
   * old way
   if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] == alleleFrequencies[ALLELE_TWO_CASE_POSITION]){
   if(alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] == alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
   riskAllele = ALLELE_ONE;
   }else if(alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] < alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
   riskAllele = ALLELE_ONE;
   }else{
   riskAllele = ALLELE_TWO;
   }
   }else if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] > alleleFrequencies[ALLELE_TWO_CASE_POSITION]){
   if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] >= alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]){
   riskAllele = ALLELE_ONE;
   }else{
   riskAllele = ALLELE_TWO;
   }
   }else{
   if(alleleFrequencies[ALLELE_TWO_CASE_POSITION] >= alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
   riskAllele = ALLELE_TWO;
   }else{
   riskAllele = ALLELE_ONE;
   }
   }*/

  //This is how Geisa does it
  if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] > alleleFrequencies[ALLELE_TWO_CASE_POSITION]){
    if(alleleFrequencies[ALLELE_ONE_CONTROL_POSITION] > alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
      if(alleleFrequencies[ALLELE_ONE_CASE_POSITION] > alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]){
        riskAllele = ALLELE_ONE;
      }else{
        riskAllele = ALLELE_TWO;
      }
    }else{
      riskAllele = ALLELE_TWO;
    }
  }else{
    if(alleleFrequencies[ALLELE_TWO_CONTROL_POSITION] > alleleFrequencies[ALLELE_ONE_CONTROL_POSITION]){
      if(alleleFrequencies[ALLELE_TWO_CASE_POSITION] > alleleFrequencies[ALLELE_TWO_CONTROL_POSITION]){
        riskAllele = ALLELE_TWO;
      }else{
        riskAllele = ALLELE_ONE;
      }
    }else{
      riskAllele = ALLELE_ONE;
    }
  }

  snp.setRiskAllele(riskAllele);
}

void BedReader::setSNPInclude(SNP& snp, const AlleleStatistics& alleleStatistics) const {
  const std::vector<double>& alleleFrequencies = alleleStatistics.getAlleleFrequencies();

  //Calculate MAF
  double minorAlleleFrequency;
  if(alleleFrequencies[ALLELE_ONE_ALL_POSITION] == alleleFrequencies[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = alleleFrequencies[ALLELE_ONE_ALL_POSITION];
  }else if(alleleFrequencies[ALLELE_ONE_ALL_POSITION] > alleleFrequencies[ALLELE_TWO_ALL_POSITION]){
    minorAlleleFrequency = alleleFrequencies[ALLELE_TWO_ALL_POSITION];
  }else{
    minorAlleleFrequency = alleleFrequencies[ALLELE_ONE_ALL_POSITION];
  }

  if(minorAlleleFrequency < minorAlleleFrequencyThreshold){
    snp.setInclude(LOW_MAF);
    return;
  }
}

} /* namespace FileIO */
} /* namespace CuEira */
