#ifndef PLINKREADERFACTORY_H_
#define PLINKREADERFACTORY_H_

#include <stdexcept>

#include <../../programOptions/Configuration.h>
#include <PlinkReader.h>
#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PlinkReaderFactory {
public:
  PlinkReaderFactory();
  virtual ~PlinkReaderFactory();

  PlinkReader* constructPlinkReader(Configuration& configuration);
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* PLINKREADERFACTORY_H_ */
