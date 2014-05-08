#include "PlinkReader.h"

namespace CuEira {
namespace FileIO {

PlinkReader::PlinkReader(BedReader& bedReader, BimReader& bimReader, FamReader& famReader) :
    bedReader(bedReader), bimReader(bimReader), famReader(famReader) {

}

PlinkReader::~PlinkReader() {

}

Container::HostVector* PlinkReader::readSNP(SNP& snp) const {
  return bedReader.readSNP(snp);
}

const PersonHandler& PlinkReader::getPersonHandler() const {
  return famReader.getPersonHandler();
}

} /* namespace FileIO */
} /* namespace CuEira */
