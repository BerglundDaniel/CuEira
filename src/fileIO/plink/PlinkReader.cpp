#include "PlinkReader.h"

namespace CuEira {
namespace FileIO {

PlinkReader::PlinkReader(BedReader& bedReader, BimReader& bimReader, FamReader& famReader) :
    bedReader(bedReader), bimReader(bimReader), famReader(famReader) {

}

PlinkReader::~PlinkReader() {

}

Container::HostVector& PlinkReader::readSNP(SNP& snp) {
  return bedReader.readSNP(snpid);
}

const PersonHandler& PlinkReader::getPersonHandler() const {
  return famReader.getPersonHandler();
}

} /* namespace FileIO */
} /* namespace CuEira */
