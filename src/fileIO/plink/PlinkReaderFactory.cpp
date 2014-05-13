#include "PlinkReaderFactory.h"

namespace CuEira {
namespace FileIO {

PlinkReaderFactory::PlinkReaderFactory() {

}

PlinkReaderFactory::~PlinkReaderFactory() {

}

PlinkReader* PlinkReaderFactory::constructPlinkReader(Configuration& configuration) {
  BimReader* bimReader = new BimReader(configuration);
  PersonHandler* personHandler = new PersonHandler();
  FamReader* famReader = new FamReader(configuration, personHandler);

  BedReader* bedReader = new BedReader(configuration, famReader->getPersonHandler(), bimReader->getNumberOfSNPs());

  return new PlinkReader(bedReader, bimReader, famReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
