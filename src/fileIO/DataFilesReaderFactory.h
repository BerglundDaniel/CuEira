#ifndef DATAFILESREADERFACTORY_H_
#define DATAFILESREADERFACTORY_H_

#include <stdexcept>
#include <map>

#include <DataFilesReader.h>
#include <Configuration.h>
#include <PlinkReaderFactory.h>
#include <PlinkReader.h>
#include <DimensionMismatch.h>
#include <CSVReader.h>
#include <EnvironmentCSVReader.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReaderFactory {
public:
  explicit DataFilesReaderFactory(PlinkReaderFactory& plinkReaderFactory);
  virtual ~DataFilesReaderFactory();

  DataFilesReader* constructDataFilesReader(Configuration& configuration);

private:
  PlinkReaderFactory& plinkReaderFactory;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADERFACTORY_H_ */
