#include "CudaContingencyTableFactory.h"

namespace CuEira {
namespace CUDA {

CudaContingencyTableFactory::CudaContingencyTableFactory(const Stream& stream) :
    stream(stream){

}

CudaContingencyTableFactory::~CudaContingencyTableFactory(){

}

const ContingencyTable* CudaContingencyTableFactory::constructContingencyTable(
    const Container::SNPVector<Container::DeviceVector>& snpVector,
    const Container::EnvironmentVector<Container::DeviceVector>& environmentVector,
    const Container::PhenotypeVector<Container::DeviceVector>& phenotypeVector) const{

  const Container::DeviceVector& snpData = snpVector.getSNPData();
  const Container::DeviceVector& envData = environmentVector.getEnvironmentData()();
  const Container::DeviceVector& phenotypeData = phenotypeVector.getPhenotypeData();
  std::vector<int>* tableCellNumbers = new std::vector<int>(8);

  const Container::DeviceMatrix* contingencyTableBlockDevice = Kernel::calculateContingencyTable(stream, snpData,
      envData, phenotypeData);

  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(0), &((*tableCellNumbers)[0]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(1), &((*tableCellNumbers)[1]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(2), &((*tableCellNumbers)[2]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(3), &((*tableCellNumbers)[3]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(4), &((*tableCellNumbers)[4]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(5), &((*tableCellNumbers)[5]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(6), &((*tableCellNumbers)[6]));
  Kernel: absoluteSumToHost(stream, (*contingencyTableBlockHost)(7), &((*tableCellNumbers)[7]));

  delete contingencyTableBlockDevice;

  return new ContingencyTable(tableCellNumbers);
}

} /* namespace CUDA */
} /* namespace CuEira */
