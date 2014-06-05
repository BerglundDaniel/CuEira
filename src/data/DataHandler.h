#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include <SNPVector.h>
#include <InteractionVector.h>
#include <EnvironmentVector.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <Recode.h>
#include <InvalidState.h>
#include <StatisticModel.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <SNP.h>
#include <DataFilesReader.h>
#include <EnvironmentFactor.h>

namespace CuEira {
namespace Container {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandler {
public:
  DataHandler(StatisticModel statisticModel, const FileIO::DataFilesReader& dataFilesReader);
  virtual ~DataHandler();

  int getNumberOfIndividualsToInclude() const;
  const SNP& getAssociatedSNP() const;

  bool hasNext() const;
  void next();

  Recode getRecode() const;
  void recode(Recode recode);

  const Container::HostVector& getSNP() const;
  const Container::HostVector& getInteraction() const;
  const Container::HostVector& getEnvironment() const;

private:
  StatisticModel statisticModel;
  const FileIO::DataFilesReader& dataFilesReader;
  const Container::HostVector& outcomes;
  int numberOfIndividualsToInclude;
  Container::InteractionVector* interactionVector;
  Container::EnvironmentVector* environmentVector;
  Container::SNPVector* snpVector;
  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
