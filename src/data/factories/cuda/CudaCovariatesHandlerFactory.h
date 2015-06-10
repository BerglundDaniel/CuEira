#ifndef CUDACOVARIATESHANDLERFACTORY_H_
#define CUDACOVARIATESHANDLERFACTORY_H_

#include <string>
#include <vector>

#include <Configuration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceMatrix.h>
#include <DeviceVector.h>
#include <CovariatesHandler.h>
#include <HostToDevice.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaCovariatesHandlerFactory {
public:
  CudaCovariatesHandlerFactory(const Configuration& configuration);
  virtual ~CudaCovariatesHandlerFactory();

  virtual CovariatesHandler<DeviceMatrix>* constructCovariatesHandler(const Stream& stream,
      const Container::PinnedHostMatrix& matrix, const std::vector<std::string>& columnNames) const;

private:
  const std::string environmentColumnName;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDACOVARIATESHANDLERFACTORY_H_ */
