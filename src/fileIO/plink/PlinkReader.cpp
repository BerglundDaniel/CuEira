#include "PlinkReader.h"

namespace CuEira {
namespace FileIO {

PlinkReader::PlinkReader(BedReader& bedReader, BimReader& bimReader, FamReader& famReader) :
    bedReader(bedReader), bimReader(bimReader), famReader(famReader) {

}

PlinkReader::~PlinkReader() {

}

Container::HostVector PlinkReader::readSNP(SNP& snp) {
  return bedReader.readSNP(snpid);
}

const Container::HostVector& PlinkReader::getOutcomes() {
  return famReader.getOutcomes();
}

int PlinkReader::getNumberOfIndividuals() {
  return famReader.getNumberOfIndividuals();
}

std::map<Id, Person>& PlinkReader::getIdToPersonMap() {
  return famReader.getIdToPersonMap();
}

} /* namespace FileIO */
} /* namespace CuEira */
