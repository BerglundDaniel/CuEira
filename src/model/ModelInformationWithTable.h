#ifndef MODELINFORMATIONWITHTABLE_H_
#define MODELINFORMATIONWITHTABLE_H_

#include <ostream>

#include <ModelInformation.h>
#include <ContingencyTable.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatistics.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelInformationWithTable: public ModelInformation {
  friend std::ostream& operator<<(std::ostream& os, const ModelInformationWithTable& modelInformationWithTable);
public:
  explicit ModelInformationWithTable(const SNP& snp, const EnvironmentFactor& environmentFactor,
      const AlleleStatistics& alleleStatistics, const ContingencyTable& contingencyTable);
  virtual ~ModelInformationWithTable();

  ModelInformationWithTable(const ModelInformationWithTable&) = delete;
  ModelInformationWithTable(ModelInformationWithTable&&) = delete;
  ModelInformationWithTable& operator=(const ModelInformationWithTable&) = delete;
  ModelInformationWithTable& operator=(ModelInformationWithTable&&) = delete;

protected:
  explicit ModelInformationWithTable(); //For the mock
  virtual void toOstream(std::ostream& os) const;

private:
  const ContingencyTable* contingencyTable;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIONWITHTABLE_H_ */
