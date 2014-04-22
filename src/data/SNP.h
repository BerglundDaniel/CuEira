#ifndef SNP_H_
#define SNP_H_

#include <Id.h>

namespace CuEira {

/**
 * This class contains information about a column of SNPs, its id and if it should be included in the calculations.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNP {
public:
  explicit SNP(Id id, bool include = true);
  virtual ~SNP();

  Id getId();
  bool getInclude();
  void setInclude(bool include);

private:
  Id id;
  bool include;
};

} /* namespace CuEira */

#endif /* SNP_H_ */
