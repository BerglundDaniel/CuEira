#ifndef DATAFILESREADER_H_
#define DATAFILESREADER_H_

#include <string>
#include <set>
#include <memory>

#include <CSVReader.h>
#include <BedReader.h>
#include <BimReader.h>
#include <Configuration.h>
#include <HostVector.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <PersonHandlerLocked.h>
#include <FileReaderException.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class DataFilesReader {
public:
  explicit DataFilesReader(std::shared_ptr<const PersonHandlerLocked> personHandler,
      std::shared_ptr<const BimReader> bimReader, std::shared_ptr<const CSVReader> csvReader,
      BedReader<Vector>* bedReader);
  virtual ~DataFilesReader();

  virtual Container::HostMatrix* readCSV() const;
  virtual const std::vector<std::string>& getCSVDataColumnNames() const;
  virtual std::vector<SNP*>* readSNPInformation() const;
  virtual Container::SNPVector<Vector>* readSNP(SNP& snp);

  virtual const PersonHandlerLocked& getPersonHandler() const;

private:
  std::shared_ptr<const PersonHandlerLocked> personHandler;
  std::shared_ptr<const BimReader> bimReader;
  std::shared_ptr<const CSVReader> csvReader;

  BedReader<Vector>* bedReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
