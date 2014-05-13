#include "PlinkReader.h"

namespace CuEira {
namespace FileIO {

PlinkReader::PlinkReader(BedReader* bedReader, BimReader* bimReader, FamReader* famReader) :
    bedReader(bedReader), bimReader(bimReader), famReader(famReader) {

}

PlinkReader::~PlinkReader() {
  delete bimReader;
  delete bedReader;
  delete famReader;
}

Container::SNPVector* PlinkReader::readSNP(SNP& snp) const {
  return bedReader->readSNP(snp);
}

const PersonHandler& PlinkReader::getPersonHandler() const {
  return famReader->getPersonHandler();
}

std::vector<SNP*> PlinkReader::getSNPInformation() {
  return bimReader->getSNPInformation();
}

} /* namespace FileIO */
} /* namespace CuEira */
