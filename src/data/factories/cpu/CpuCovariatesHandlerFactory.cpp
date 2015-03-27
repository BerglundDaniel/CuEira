#include "CpuCovariatesHandlerFactory.h"

namespace CuEira {
namespace CPU {

CpuCovariatesHandlerFactory::CpuCovariatesHandlerFactory(const Configuration& configuration,
    const MKLWrapper& mklWrapper) :
    environmentColumnName(configuration.getEnvironmentColumnName()), mklWrapper(mklWrapper) {

}

CpuCovariatesHandlerFactory::~CpuCovariatesHandlerFactory() {

}

CovariatesHandler<HostMatrix>* CpuCovariatesHandlerFactory::constructCovariatesHandler(
    const Container::HostMatrix& covariates, const std::vector<std::string>& columnNames) const {
  const int numberOfColumns = covariates.getNumberOfColumns();
  Container::RegularHostMatrix* covariatesDevice = new Container::RegularHostMatrix(covariates.getNumberOfRows(),
      numberOfColumns - 1);

  int col = 0;
  for(int i = 0; i < numberOfColumns; ++i){
    if(environmentColumnName != columnNames[i]){
      Container::HostVector* covVectorFrom = covariates(i);
      Container::HostVector* covVectorTo = covariates(i);

      mklWrapper.copyVector(covVectorFrom, covVectorTo);

      delete covVectorFrom;
      delete covVectorFrom;
      ++col;
    }
  }

  return new CovariatesHandler<HostMatrix>(covariates);
}

} /* namespace CPU */
} /* namespace CuEira */
