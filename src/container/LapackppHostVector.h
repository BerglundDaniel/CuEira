#ifndef LAPACKPPHOSTVECTOR_H_
#define LAPACKPPHOSTVECTOR_H_

#include <lapackpp/lavd.h>
#include <lapackpp/laexcp.h>

#include <HostVector.h>
#include <DimensionMismatch.h>
#include <LapackppHostMatrix.h>

namespace CuEira {
namespace Container {
class LapackppHostMatrix;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostVector: public HostVector {
  friend LapackppHostMatrix;
public:
  LapackppHostVector(LaVectorDouble* lapackppContainer);
  virtual ~LapackppHostVector();

  LaVectorDouble& getLapackpp();
  virtual double& operator()(unsigned int index);
  virtual const double& operator()(unsigned int index) const;

protected:
  LapackppHostVector(LaVectorDouble* lapackppContainer, bool subview);

private:
  LaVectorDouble* lapackppContainer;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* LAPACKPPHOSTVECTOR_H_ */
