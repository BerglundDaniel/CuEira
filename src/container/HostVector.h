#ifndef HOSTVECTOR_H_
#define HOSTVECTOR_H_

namespace CuEira {
namespace CUDA {
class DeviceToHost;
class HostToDevice;
}
namespace Container {


/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class HostVector {
  friend CUDA::DeviceToHost;
  friend CUDA::HostToDevice;
public:
  HostVector(unsigned int numberOfRows, bool subview, PRECISION* hostVector);
  virtual ~HostVector();

  int getNumberOfRows() const;
  int getNumberOfColumns() const;
  virtual PRECISION& operator()(unsigned int index)=0;
  virtual const PRECISION& operator()(unsigned int index) const=0;

protected:
  PRECISION* getMemoryPointer();
  const PRECISION* getMemoryPointer() const;

  PRECISION* hostVector;
  const unsigned int numberOfRows;
  const unsigned int numberOfColumns;
  const bool subview;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTVECTOR_H_ */
