#ifndef PINNEDHOSTMATRIX_H_
#define PINNEDHOSTMATRIX_H_

#include "HostMatrix.h"

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostMatrix: public HostMatrix {
public:
  PinnedHostMatrix();
  virtual ~PinnedHostMatrix();
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTMATRIX_H_ */
