#ifndef CONTINGENCYTABLE_H_
#define CONTINGENCYTABLE_H_

#include <ostream>
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
  friend std::ostream& operator<<(std::ostream& os, const ContingencyTable& contingencyTable);
public:
  ContingencyTable(const std::vector<int>* tableCellNumbers);
  virtual ~ContingencyTable();

  virtual const std::vector<int>& getTable() const;

  ContingencyTable(const ContingencyTable&) = delete;
  ContingencyTable(ContingencyTable&&) = delete;
  ContingencyTable& operator=(const ContingencyTable&) = delete;
  ContingencyTable& operator=(ContingencyTable&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const;

private:
  const std::vector<int>* tableCellNumbers;
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLE_H_ */
