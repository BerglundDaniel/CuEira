#include "PlinkReaderFactory.h"

namespace CuEira {
namespace FileIO {

PlinkReaderFactory::PlinkReaderFactory() {

}

PlinkReaderFactory::~PlinkReaderFactory() {

}

PlinkReader* PlinkReaderFactory::constructPlinkReader(Configuration& configuration) {
  BimReader bimReader(configuration);
  PersonHandler* personHandler = new PersonHandler();
  FamReader famReader(configuration, personHandler);

  BedReader bedReader(configuration, famReader.getPersonHandler(), bimReader.getNumberOfSNPs());

  return new PlinkReader(bedReader, bimReader, famReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
