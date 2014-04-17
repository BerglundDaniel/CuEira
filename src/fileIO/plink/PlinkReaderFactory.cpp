#include "PlinkReaderFactory.h"

namespace CuEira {
namespace FileIO {

PlinkReaderFactory::PlinkReaderFactory() {

}

PlinkReaderFactory::~PlinkReaderFactory() {

}

PlinkReader* PlinkReaderFactory::constructPlinkReader(Configuration& configuration) {
  BedReader bedReader = BedReader(configuration);
  BimReader bimReader = BimReader(configuration);
  FamReader fameReader = FamReader(configuration);

  return new PlinkReader(bedReader, bimReader, famReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
