#ifndef LAPACKPPHOSTMATRIX_H_
#define LAPACKPPHOSTMATRIX_H_

#include <lapackpp/gmd.h>
#include <lapackpp/lavd.h>
#include <lapackpp/laexcp.h>

#include <HostMatrix.h>
#include <HostVector.h>
#include <LapackppHostVector.h>
#include <DimensionMismatch.h>

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
  const LaGenMatDouble& getLapackpp() const;
  virtual HostVector* operator()(unsigned int column);
  virtual const HostVector* operator()(unsigned int column) const;
  virtual double& operator()(unsigned int row, unsigned int column);
  virtual const double& operator()(unsigned int row, unsigned int column) const;

  LapackppHostMatrix(const LapackppHostMatrix&) = delete;
  LapackppHostMatrix(LapackppHostMatrix&&) = delete;
  LapackppHostMatrix& operator=(const LapackppHostMatrix&) = delete;
  LapackppHostMatrix& operator=(LapackppHostMatrix&&) = delete;

private:
  LaGenMatDouble* lapackppContainer;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* LAPACKPPHOSTMATRIX_H_ */
