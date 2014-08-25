#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_

#include <utility>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

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

namespace CuEira {
class DataHandlerTest;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandler {
  friend DataHandlerTest;
  FRIEND_TEST(DataHandlerTest, Next);
  FRIEND_TEST(DataHandlerTest, Recode);
public:
  DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
      const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
      Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector);
  virtual ~DataHandler();

  virtual const SNP& getCurrentSNP() const;
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;

  virtual DataHandlerState next();

  virtual Recode getRecode() const;
  virtual void recode(Recode recode);

  virtual const Container::SNPVector& getSNPVector() const;
  virtual const Container::InteractionVector& getInteractionVector() const;
  virtual const Container::EnvironmentVector& getEnvironmentVector() const;

protected:
  DataHandler(); //For the mock

private:
  enum State {
    NOT_INITIALISED, INITIALISED
  };

  bool readSNP(SNP& nextSnp);

  State state;
  Task::DataQueue* dataQueue;
  const StatisticModel statisticModel;
  const FileIO::BedReader* bedReader;
  const std::vector<const EnvironmentFactor*>* environmentInformation;
  Container::EnvironmentVector* environmentVector;
  Container::SNPVector* snpVector;
  Container::InteractionVector* interactionVector;
  Recode currentRecode;
  int currentEnvironmentFactorPos;
};

} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
