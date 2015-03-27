#ifndef CPUCOVARIATESHANDLERFACTORY_H_
#define CPUCOVARIATESHANDLERFACTORY_H_

#include <string>
#include <vector>

#include <Configuration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <CovariatesHandler.h>
#include <MKLWrapper.h>

namespace CuEira {
namespace CPU {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuCovariatesHandlerFactory {
public:
  CpuCovariatesHandlerFactory(const Configuration& configuration, const MKLWrapper& mklWrapper);
  virtual ~CpuCovariatesHandlerFactory();

  virtual CovariatesHandler<HostMatrix>* constructCovariatesHandler(const Container::HostMatrix& covariates,
      const std::vector<std::string>& columnNames) const;

private:
  const std::string environmentColumnName;
  const MKLWrapper& mklWrapper;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUCOVARIATESHANDLERFACTORY_H_ */
