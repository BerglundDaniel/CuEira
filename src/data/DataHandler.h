#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include <utility>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <Configuration.h>
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
#include <DataHandlerState.h>
#include <ContingencyTable.h>
#include <ContingencyTableFactory.h>
#include <Configuration.h>
#include <AlleleStatistics.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>

namespace CuEira {
class DataHandlerTest;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandler {
  friend DataHandlerTest;FRIEND_TEST(DataHandlerTest, Next);FRIEND_TEST(DataHandlerTest, Recode);
public:
  DataHandler(const Configuration& configuration, const FileIO::BedReader& bedReader,
      const ContingencyTableFactory& contingencyTableFactory,
      const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
      Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector);
  virtual ~DataHandler();

  virtual const SNP& getCurrentSNP() const;
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;
  virtual const ContingencyTable& getContingencyTable() const;
  virtual const AlleleStatistics& getAlleleStatistics() const;

  virtual DataHandlerState next();

  virtual Recode getRecode() const;
  virtual void recode(Recode recode);

  virtual const Container::SNPVector& getSNPVector() const;
  virtual const Container::InteractionVector& getInteractionVector() const;
  virtual const Container::EnvironmentVector& getEnvironmentVector() const;

protected:
  DataHandler(const Configuration& configuration); //For the mock

private:
  enum State {
    NOT_INITIALISED, INITIALISED
  };

  bool readSNP(SNP& nextSnp);
  void setSNPInclude(SNP& snp, const ContingencyTable& contingencyTable) const;

  const Configuration& configuration;
  State state;
  const ContingencyTableFactory* contingencyTableFactory;
  Task::DataQueue* dataQueue;
  const StatisticModel statisticModel;
  const FileIO::BedReader* bedReader;
  const std::vector<const EnvironmentFactor*>* environmentInformation;
  Container::EnvironmentVector* environmentVector;
  Container::SNPVector* snpVector;
  Container::InteractionVector* interactionVector;
  const ContingencyTable* contingencyTable;
  const AlleleStatistics* alleleStatistics;
  Recode currentRecode;
  int currentEnvironmentFactorPos;
  SNP* currentSNP;
  const int cellCountThreshold;
};

} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
