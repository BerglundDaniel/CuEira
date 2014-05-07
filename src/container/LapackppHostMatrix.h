#ifndef LAPACKPPHOSTMATRIX_H_
#define LAPACKPPHOSTMATRIX_H_

#include <lapackpp/gmd.h>
#include <lapackpp/lavd.h>
#include <lapackpp/laexcp.h>

#include <HostMatrix.h>
#include <HostVector.h>
#include <LapackppHostVector.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LapackppHostMatrix: public HostMatrix {
public:
  LapackppHostMatrix(LaGenMatDouble* lapackppContainer);
  virtual ~LapackppHostMatrix();

  LaGenMatDouble& getLapackpp();
  virtual HostVector* operator()(int column);
  virtual const HostVector* operator()(int column) const;
  virtual double& operator()(int row, int column);
  virtual const double& operator()(int row, int column) const;
private:
  LaGenMatDouble* lapackppContainer;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* LAPACKPPHOSTMATRIX_H_ */
