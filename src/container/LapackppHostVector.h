#ifndef LAPACKPPHOSTVECTOR_H_
#define LAPACKPPHOSTVECTOR_H_

#include <lapackpp/lavd.h>
#include <lapackpp/laexcp.h>

#include <HostVector.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostVector: public HostVector {
public:
  LapackppHostVector(LaVectorDouble* lapackppContainer);
  LapackppHostVector(LaVectorDouble* lapackppContainer, bool subview);
  virtual ~LapackppHostVector();

  LaVectorDouble& getLapackpp();
  virtual double& operator()(int index);
  virtual const double& operator()(int index) const;

private:
  LaVectorDouble* lapackppContainer;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* LAPACKPPHOSTVECTOR_H_ */
