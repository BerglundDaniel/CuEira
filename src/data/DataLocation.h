#ifndef DATALOCATION_H_
#define DATALOCATION_H_

namespace CuEira {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class DataLocation {
public:
  explicit DataLocation(Vector* snpVector, Vector* environmentVector, Vector* interactionVector,
      Vector* phenotypeVector, Matrix* covariatesMatrix);
  virtual ~DataLocation();

  Vector& getSnpVector();
  Vector& getEnvironmentVector();
  Vector& getInteractionVector();
  Vector& getPhenotypeVector();
  Matrix& getCovariatesMatrix();

  DataLocation(const DataLocation&) = delete;
  DataLocation(DataLocation&&) = delete;
  DataLocation& operator=(const DataLocation&) = delete;
  DataLocation& operator=(DataLocation&&) = delete;

protected:
  Vector* snpVector;
  Vector* environmentVector;
  Vector* interactionVector;
  Vector* phenotypeVector;
  Matrix* covariatesMatrix;

};

} /* namespace CuEira */

#endif /* DATALOCATION_H_ */
