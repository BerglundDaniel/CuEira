#include "DataFilesReader.h"

namespace CuEira {
namespace FileIO {

DataFilesReader::DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
    CSVReader* csvReader) :
    personHandler(personHandler), bedReader(bedReader), bimReader(bimReader), csvReader(csvReader) {
  personHandler->lockIndividuals();
}

DataFilesReader::~DataFilesReader() {
  delete bedReader;
  delete bimReader;
  delete csvReader;
  delete personHandler;
}

Container::HostMatrix* DataFilesReader::readCSV() const {
  return csvReader->readData();
}

const std::vector<std::string>& DataFilesReader::getCSVDataColumnNames() const {
  return csvReader->getDataColumnNames();
}

std::vector<SNP*>* DataFilesReader::readSNPInformation() const {
  return bimReader->readSNPInformation();
}

Container::SNPVector* DataFilesReader::readSNP(SNP& snp) {
  return bedReader->readSNP(snp);
}

const PersonHandler& DataFilesReader::getPersonHandler() const {
  return *personHandler;
}

} /* namespace FileIO */
} /* namespace CuEira */
