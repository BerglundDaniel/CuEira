#ifndef PHENOTYPEHANDLER_H_
#define PHENOTYPEHANDLER_H_

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PhenotypeHandler {
public:
  virtual ~PhenotypeHandler();

  virtual int getNumberOfIndividuals() const;

protected:
  explicit PhenotypeHandler(int numberOfIndividuals);

  const int numberOfIndividuals;
};

} /* namespace CuEira */

#endif /* PHENOTYPEHANDLER_H_ */
