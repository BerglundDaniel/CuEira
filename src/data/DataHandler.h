#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include <utility>

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
#include <BedReader.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <DataQueue.h>

namespace CuEira {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandler {
public:
  DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
      const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue);
  virtual ~DataHandler();

  int getNumberOfIndividualsToInclude() const;
  const SNP& getCurrentSNP() const;
  const EnvironmentFactor& getCurrentEnvironmentFactor() const;

  bool next();

  Recode getRecode() const;
  void recode(Recode recode);

  const Container::HostVector& getSNP() const;
  const Container::HostVector& getInteraction() const;
  const Container::HostVector& getEnvironment() const;

private:
  void readSNP(SNP& nextSnp);

  Task::DataQueue& dataQueue;
  const StatisticModel statisticModel;
  const FileIO::BedReader& bedReader;
  const EnvironmentFactorHandler& environmentFactorHandler;
  const std::vector<EnvironmentFactor*>& environmentInformation;
  int numberOfIndividualsToInclude;
  Container::EnvironmentVector* environmentVector;
  Container::SNPVector* snpVector;
  Container::InteractionVector* interactionVector;
  Recode currentRecode;
  bool firstNext;
  SNP* currentSNP;
  int currentEnvironmentFactorPos;
};

} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
