#include "PlinkReaderFactory.h"

namespace CuEira {
namespace FileIO {

PlinkReaderFactory::PlinkReaderFactory() {

}

PlinkReaderFactory::~PlinkReaderFactory() {

}

PlinkReader* PlinkReaderFactory::constructPlinkReader(Configuration& configuration) {
  BimReader bimReader(configuration);
  FamReader famReader(configuration);
  BedReader bedReader(configuration, famReader.getOutcomes(), famReader.getNumberOfIndividuals(),
      bimReader.getNumberOfSNPs());

  return new PlinkReader(bedReader, bimReader, famReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
