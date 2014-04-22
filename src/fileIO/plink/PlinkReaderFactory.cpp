#include "PlinkReaderFactory.h"

namespace CuEira {
namespace FileIO {

PlinkReaderFactory::PlinkReaderFactory() {

}

PlinkReaderFactory::~PlinkReaderFactory() {

}

PlinkReader* PlinkReaderFactory::constructPlinkReader(Configuration& configuration) {
  BedReader bedReader(configuration);
  BimReader bimReader(configuration);
  FamReader fameReader(configuration);

  return new PlinkReader(bedReader, bimReader, famReader);
}

} /* namespace FileIO */
} /* namespace CuEira */
