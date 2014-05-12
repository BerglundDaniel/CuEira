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

BedReader::~BedReader() {

}

Container::LapackppHostVector* BedReader::readSNP(SNP& snp) const {
  std::ifstream bedFile;
  int numberOfAlleleOneCase = 0;
  int numberOfAlleleTwoCase = 0;
  int numberOfAlleleOneControl = 0;
  int numberOfAlleleTwoControl = 0;
  int numberOfAlleleOneAll = 0;
  int numberOfAlleleTwoAll = 0;
  const int snpPos=snp.getPosition();

  //Initialise vector
#ifdef CPU
  LaVectorDouble* laVector = new LaVectorDouble(numberOfIndividualsToInclude);
  Container::LapackppHostVector* SNPVector = new Container::LapackppHostVector(laVector);
#else
  Container::PinnedHostVector* SNPVector = new Container::PinnedHostVector(numberOfIndividualsToInclude);
#endif

  openBedFile(bedFile);

  //Read depending on the mode
  if(mode == SNPMAJOR){
    const int numberOfBitsPerRow = numberOfIndividualsTotal * 2;
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

      //We have to fix the buffersize if it's the lastread and if we couldn't read the whole row at once
      if(readNumber == numberOfReads && readNumber != 1){
        readBufferSize = numberOfBytesPerRow - readBufferSize * (numberOfReads - 1);
      }

      char buffer[readBufferSize];
      bedFile.seekg(headerSize + numberOfBytesPerRow * readNumber);   //FIXME which snp number are we actually at?
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
        for(int bitPairNumber = 1; bitPairNumber <= numberOfBitPairsPerByte; ++bitPairNumber){
          //It's in reverse due to plinks format that has the bits in each byte in reverse
          int posInByte = 8 - 2 * bitPairNumber;
          bool firstBit = getBit(currentByte, posInByte + 1);
          bool secondBit = getBit(currentByte, posInByte);
          //The position in the vector where we are going to store the geneotype for this individual
          int personRowFileNumber = (readNumber - 1) * readBufferSize + byteNumber * 4 + bitPairNumber - 1;
          const Person& person = personHandler.getPersonFromRowAll(personRowFileNumber);

          if(person.getInclude()){ //If the person shouldn't be included we will skip it
            Phenotype phenotype = person.getPhenotype();
            int currentPersonRow = personHandler.getRowIncludeFromPerson(person);

            //If we are missing the genotype for one individual(that should be included) or more we excluded the SNP
            if(firstBit && !secondBit){
              snp.setInclude(false);
              closeBedFile(bedFile);
              return SNPVector; //Since we are going to exclude this SNP there is no point in reading more data.
            }/* if check missing */

            //Store the genotype as 0,1,2 until we can recode it. We have to know the risk allele before we can recode.
            //Also increase the counters for the alleles if it is a case.
            if(!firstBit && !secondBit){
              //Homozygote primary
              (*SNPVector)(currentPersonRow) = 0;
              numberOfAlleleOneAll += 2;

              if(phenotype == AFFECTED){
                numberOfAlleleOneCase += 2;
              }else{
                numberOfAlleleOneControl += 2;
              }
            }else if(!firstBit && secondBit){
              //Hetrozygote
              (*SNPVector)(currentPersonRow) = 1;
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
              (*SNPVector)(currentPersonRow) = 2;
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
  const double alleleOneCaseFrequency = numberOfAlleleOneCase / numberOfAllelesInPopulation;
  const double alleleTwoCaseFrequency = numberOfAlleleTwoCase / numberOfAllelesInPopulation;
  const double alleleOneControlFrequency = numberOfAlleleOneControl / numberOfAllelesInPopulation;
  const double alleleTwoControlFrequency = numberOfAlleleTwoControl / numberOfAllelesInPopulation;
  const double alleleOneAllFrequency = numberOfAlleleOneAll / numberOfAllelesInPopulation;
  const double alleleTwoAllFrequency = numberOfAlleleTwoAll / numberOfAllelesInPopulation;

  snp.setCaseAlleleFrequencies(alleleOneCaseFrequency, alleleTwoCaseFrequency);
  snp.setControlAlleleFrequencies(alleleOneControlFrequency, alleleTwoControlFrequency);
  snp.setAllAlleleFrequencies(alleleOneAllFrequency, alleleTwoAllFrequency);

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
  double minorAlleleFrequency;

  if(alleleOneAllFrequency == alleleTwoAllFrequency){
    minorAlleleFrequency = alleleOneAllFrequency;
  }else if(alleleOneAllFrequency > alleleTwoAllFrequency){
    minorAlleleFrequency = alleleOneAllFrequency;
  }else{
    minorAlleleFrequency = alleleTwoAllFrequency;
  }
  snp.setMinorAlleleFrequency(minorAlleleFrequency);
  if(minorAlleleFrequencyThreshold > minorAlleleFrequency){
    snp.setInclude(false);
  }

  return SNPVector;
}
/* readSNP */

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
