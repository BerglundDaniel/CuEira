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
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <SNP.h>
#include <BedReader.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <DataQueue.h>
#include <ContingencyTable.h>
#include <ContingencyTableFactory.h>
#include <Configuration.h>
#include <DataHandlerState.h>
#include <AlleleStatistics.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>
#include <CovariatesMatrix.h>
#include <ModelInformation.h>
#include <ModelInformationFactory.h>
#include <InteractionModel.h>
#include <AlleleStatisticsFactory.h>
#include <RiskAlleleStrategy.h>

#ifdef PROFILE
#include <boost/chrono/chrono_io.hpp>
#endif

namespace CuEira {
class DataHandlerTest;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class DataHandler {
  friend DataHandlerTest;FRIEND_TEST(DataHandlerTest, Recode);FRIEND_TEST(DataHandlerTest, RecodeEnvNotBinary);FRIEND_TEST(DataHandlerTest, RecodeEnvBinary);FRIEND_TEST(DataHandlerTest, ApplyStatisticModel_PrevFalse);FRIEND_TEST(DataHandlerTest, ApplyStatisticModel_PrevTrue);
public:
  DataHandler(const Configuration& configuration, FileIO::BedReader<>& bedReader,
      const ContingencyTableFactory& contingencyTableFactory,
      const Model::ModelInformationFactory* modelInformationFactory, Task::DataQueue& dataQueue,
      Container::EnvironmentVector<Vector>* environmentVector, Container::InteractionVector<Vector>* interactionVector,
      Container::PhenotypeVector<Vector>* phenotypeVector,
      Container::CovariatesMatrix<Matrix, Vector>* covariatesMatrix, MissingDataHandler<Vector>* missingDataHandler,
      const AlleleStatisticsFactory<Vector>* alleleStatisticsFactory);
  virtual ~DataHandler();

  virtual DataHandlerState next();
  virtual void applyStatisticModel(const InteractionModel<Vector>& interactionModel);

  virtual Recode getRecode() const;
  virtual void recode(Recode recode);

  virtual const Model::ModelInformation& getCurrentModelInformation() const;

  virtual const SNP& getCurrentSNP() const;
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;

  virtual const Container::SNPVector<Vector>& getSNPVector() const;
  virtual const Container::InteractionVector<Vector>& getInteractionVector() const;
  virtual const Container::EnvironmentVector<Vector>& getEnvironmentVector() const;
  virtual const Container::CovariatesMatrix<Matrix, Vector>& getCovariatesMatrix() const;

  DataHandler(const DataHandler<Matrix, Vector>&) = delete;
  DataHandler(DataHandler<Matrix, Vector> &&) = delete;
  DataHandler<Matrix, Vector>& operator=(const DataHandler<Matrix, Vector>&) = delete;
  DataHandler<Matrix, Vector>& operator=(DataHandler<Matrix, Vector> &&) = delete;

#ifdef PROFILE
  static boost::chrono::duration<long long, boost::nano> timeSpentRecode;
  static boost::chrono::duration<long long, boost::nano> timeSpentNext;
  static boost::chrono::duration<long long, boost::nano> timeSpentSNPRead;
  static boost::chrono::duration<long long, boost::nano> timeSpentStatModel;
#endif

protected:
  DataHandler(const Configuration& configuration); //For the mock

private:
  enum State {
    NOT_INITIALISED, INITIALISED
  };

  const Configuration& configuration;
  State state;
  const ContingencyTableFactory<Vector>* contingencyTableFactory;
  const Model::ModelInformationFactory* modelInformationFactory;
  const AlleleStatisticsFactory<Vector>* alleleStatisticsFactory;
  const RiskAlleleStrategy* riskAlleleStrategy;
  Task::DataQueue* dataQueue;
  FileIO::BedReader<>* bedReader;
  MissingDataHandler<Vector>* missingDataHandler;

  Container::CovariatesMatrix<Matrix, Vector> covariatesMatrix;
  Container::EnvironmentVector<Vector>* environmentVector;
  Container::SNPVector<Vector>* snpVector;
  Container::InteractionVector<Vector>* interactionVector;
  Container::PhenotypeVector<Vector>* phenotypeVector;
  const Model::ModelInformation* modelInformation;
  const ContingencyTable* contingencyTable;
  const AlleleStatistics* alleleStatistics;

  Recode currentRecode;
  SNP* currentSNP;
  const EnvironmentFactor* environmentFactor;
  const int cellCountThreshold;
  const double minorAlleleFrequencyThreshold;
  bool appliedStatisticModel;
};

} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
