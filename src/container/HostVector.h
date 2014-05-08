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
  HostVector(unsigned int numberOfRows, bool subview, PRECISION* hostVector);
  virtual ~HostVector();

  int getNumberOfRows();
  int getNumberOfColumns();
  virtual PRECISION& operator()(unsigned int index)=0;
  virtual const PRECISION& operator()(unsigned int index) const=0;

protected:
  PRECISION* getMemoryPointer();

  PRECISION* hostVector;
  const unsigned int numberOfRows;
  const unsigned int numberOfColumns;
  const bool subview;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* HOSTVECTOR_H_ */
