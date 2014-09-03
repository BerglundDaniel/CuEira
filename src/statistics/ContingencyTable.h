#ifndef CONTINGENCYTABLE_H_
#define CONTINGENCYTABLE_H_

#include <vector>

#define SNP0_ENV0_CONTROL_POSITION 0
#define SNP1_ENV0_CONTROL_POSITION 1
#define SNP0_ENV1_CONTROL_POSITION 2
#define SNP1_ENV1_CONTROL_POSITION 3

#define SNP0_ENV0_CASE_POSITION 4
#define SNP1_ENV0_CASE_POSITION 5
#define SNP0_ENV1_CASE_POSITION 6
#define SNP1_ENV1_CASE_POSITION 7

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ContingencyTable {
public:
  ContingencyTable(const std::vector<int>* tableCellNumbers);
  virtual ~ContingencyTable();

  virtual const std::vector<int>& getTable() const;

private:
  const std::vector<int>* tableCellNumbers;
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLE_H_ */
