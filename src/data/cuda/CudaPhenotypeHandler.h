#ifndef CUDAPHENOTYPEHANDLER_H_
#define CUDAPHENOTYPEHANDLER_H_

#include <PhenotypeHandler.h>
#include <DeviceVector.h>

namespace CuEira {
namespace CUDA {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaPhenotypeHandler: public PhenotypeHandler {
public:
  explicit CudaPhenotypeHandler(const Container::DeviceVector* phenotypeData);
  virtual ~CudaPhenotypeHandler();

  virtual const Container::DeviceVector& getPhenotypeData() const;

private:
  const Container::DeviceVector* phenotypeData;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAPHENOTYPEHANDLER_H_ */
