#ifndef PINNEDHOSTVECTOR_H_
#define PINNEDHOSTVECTOR_H_

#include <HostVector.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PinnedHostVector: public HostVector {
public:
  PinnedHostVector();
  virtual ~PinnedHostVector();
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PINNEDHOSTVECTOR_H_ */
