#ifndef LAPACKPPHOSTMATRIX_H_
#define LAPACKPPHOSTMATRIX_H_

#include "HostMatrix.h"

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostMatrix: public HostMatrix {
public:
  LapackppHostMatrix();
  virtual ~LapackppHostMatrix();
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* LAPACKPPHOSTMATRIX_H_ */
