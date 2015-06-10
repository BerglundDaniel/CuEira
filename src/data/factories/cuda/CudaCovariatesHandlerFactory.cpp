#include "CudaCovariatesHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaCovariatesHandlerFactory::CudaCovariatesHandlerFactory(const Configuration& configuration) :
    environmentColumnName(configuration.getEnvironmentColumnName()){

}

CudaCovariatesHandlerFactory::~CudaCovariatesHandlerFactory(){

}

CovariatesHandler<DeviceMatrix>* CudaCovariatesHandlerFactory::constructCovariatesHandler(const Stream& stream,
    const Container::PinnedHostMatrix& matrix, const std::vector<std::string>& columnNames) const{
  const int numberOfColumns = matrix.getNumberOfColumns();
  Container::DeviceMatrix* covariatesDevice = new Container::DeviceMatrix(matrix.getNumberOfRows(),
      numberOfColumns - 1);

  int col = 0;
  for(int i = 0; i < numberOfColumns; ++i){
    if(environmentColumnName != columnNames[i]){
      const Container::PinnedHostVector* covVector = matrix(i);
      transferVector(stream, *covVector, (*covariatesDevice)(0, col));

      delete covVector;
      ++col;
    }
  }

  return new CovariatesHandler<DeviceMatrix>(covariatesDevice);
}

} /* namespace CUDA */
} /* namespace CuEira */
