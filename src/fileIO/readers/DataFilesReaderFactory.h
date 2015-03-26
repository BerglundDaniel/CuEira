#ifndef DATAFILESREADERFACTORY_H_
#define DATAFILESREADERFACTORY_H_

#include <stdexcept>
#include <map>

#include <DataFilesReader.h>
#include <Configuration.h>
#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <DimensionMismatch.h>
#include <CSVReader.h>
#include <PersonHandler.h>
#include <SNPVectorFactory.h>
#include <CovariatesHandlerFactory.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReaderFactory {
public:
  explicit DataFilesReaderFactory();
  virtual ~DataFilesReaderFactory();

  DataFilesReader* constructDataFilesReader(Configuration& configuration);

private:

};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADERFACTORY_H_ */
