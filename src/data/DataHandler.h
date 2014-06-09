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
#include <BedReader.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>

namespace CuEira {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandler {
public:
  DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader, const EnvironmentFactorHandler& environmentFactorHandler);
  virtual ~DataHandler();

  int getNumberOfIndividualsToInclude() const;
  const SNP& getAssociatedSNP() const;

  bool next();

  Recode getRecode() const;
  void recode(Recode recode);

  const Container::HostVector& getSNP() const;
  const Container::HostVector& getInteraction() const;
  const Container::HostVector& getEnvironment() const;
  const Container::HostMatrix& getCovariates() const; //TODO

private:
  StatisticModel statisticModel;
  const FileIO::BedReader& bedReader;
  const EnvironmentFactorHandler& environmentFactorHandler;
  int numberOfIndividualsToInclude;
  Container::EnvironmentVector* environmentVector;
  Container::SNPVector* snpVector;
  Container::InteractionVector* interactionVector;
  Recode currentRecode;
};

} /* namespace CuEira */

#endif /* DATAHANDLER_H_ */
