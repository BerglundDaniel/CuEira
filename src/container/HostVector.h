#ifndef HOSTVECTOR_H_
#define HOSTVECTOR_H_

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class HostVector {
public:
  HostVector(int numberOfRows, bool subview, PRECISION* hostVector);
  virtual ~HostVector();

  int getNumberOfRows();
  int getNumberOfColumns();
  virtual PRECISION& operator()(int index)=0;
  virtual const PRECISION& operator()(int index) const=0;

protected:
  PRECISION* getMemoryPointer();

  PRECISION* hostVector;
  const int numberOfRows;
  const int numberOfColumns;
  const bool subview;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTVECTOR_H_ */
