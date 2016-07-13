#ifndef DATALOCATION_H_
#define DATALOCATION_H_

namespace CuEira {

using namespace CuEira::Container;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class DataLocation {
public:
  explicit DataLocation(Vector& snpVector, Vector& environmentVector, Vector& interactionVector,
      Vector& phenotypeVector, Matrix& covariatesMatrix);
  virtual ~DataLocation();

  Vector& snpVector;
  Vector& environmentVector;
  Vector& interactionVector;
  Vector& phenotypeVector;
  Matrix& covariatesMatrix;

  DataLocation(const DataLocation&) = delete;
  DataLocation(DataLocation&&) = delete;
  DataLocation& operator=(const DataLocation&) = delete;
  DataLocation& operator=(DataLocation&&) = delete;
};

} /* namespace CuEira */

#endif /* DATALOCATION_H_ */
