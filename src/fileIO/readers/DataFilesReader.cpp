#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

template<typename Vector>
DataFilesReader<Vector>::DataFilesReader(std::shared_ptr<const PersonHandlerLocked> personHandler,
    std::shared_ptr<const BimReader> bimReader, std::shared_ptr<const CSVReader> csvReader,
    BedReader<Vector>* bedReader) :
    personHandler(personHandler), bedReader(bedReader), bimReader(bimReader), csvReader(csvReader){

}

template<typename Vector>
DataFilesReader<Vector>::~DataFilesReader(){
  delete bedReader;
}

template<typename Vector>
Container::HostMatrix* DataFilesReader<Vector>::readCSV() const{
  return csvReader->readData(personHandler);
}

template<typename Vector>
const std::vector<std::string>& DataFilesReader<Vector>::getCSVDataColumnNames() const{
  return csvReader->getDataColumnNames();
}

template<typename Vector>
std::vector<SNP*>* DataFilesReader<Vector>::readSNPInformation() const{
  return bimReader->readSNPInformation();
}

template<typename Vector>
Container::SNPVector<Vector>* DataFilesReader<Vector>::readSNP(SNP& snp){
  return bedReader->readSNP(snp);
}

template<typename Vector>
const PersonHandlerLocked& DataFilesReader<Vector>::getPersonHandler() const{
  return *personHandler;
}

} /* namespace FileIO */
} /* namespace CuEira */
