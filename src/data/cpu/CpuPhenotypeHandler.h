#ifndef CPUPHENOTYPEHANDLER_H_
#define CPUPHENOTYPEHANDLER_H_

#include <PhenotypeHandler.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace CPU {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuPhenotypeHandler: public PhenotypeHandler {
public:
  explicit CpuPhenotypeHandler(const Container::RegularHostVector* phenotypeData);
  virtual ~CpuPhenotypeHandler();

  virtual const Container::RegularHostVector& getPhenotypeData() const;

private:
  const Container::RegularHostVector* phenotypeData;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUPHENOTYPEHANDLER_H_ */
