#ifndef CPUCOVARIATESHANDLERFACTORY_H_
#define CPUCOVARIATESHANDLERFACTORY_H_

#include <string>
#include <vector>

#include <Configuration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <RegularHostMatrix.h>
#include <RegularHostVector.h>
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
  CpuCovariatesHandlerFactory(const Configuration& configuration);
  virtual ~CpuCovariatesHandlerFactory();

  virtual CovariatesHandler<Container::HostMatrix>* constructCovariatesHandler(const Container::HostMatrix& matrix,
      const std::vector<std::string>& columnNames) const;

private:
  const std::string environmentColumnName;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUCOVARIATESHANDLERFACTORY_H_ */
