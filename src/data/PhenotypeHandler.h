#ifndef PHENOTYPEHANDLER_H_
#define PHENOTYPEHANDLER_H_

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class PhenotypeHandler {
public:
  explicit PhenotypeHandler(const Vector* vector);
  virtual ~PhenotypeHandler();

  virtual int getNumberOfIndividualsTotal() const;
  virtual const Vector& getPhenotypeData() const;

private:
  const Vector* vector;
  const int numberOfIndividualsTotal;
};

} /* namespace CuEira */

#endif /* PHENOTYPEHANDLER_H_ */
