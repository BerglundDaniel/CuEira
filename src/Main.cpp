#include <iostream>
#include <stdexcept>
#include <vector>

#include <Configuration.h>
#include <PlinkReaderFactory.h>
#include <PlinkReader.h>
#include <DataFilesReaderFactory.h>
#include <DataFilesReader.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <PersonHandler.h>
#include <EnvironmentFactor.h>
#include <Statistics.h>
#include <Recode.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#include <LogisticRegression.h>
#include <MultipleLogisticRegression.h>
#else
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#endif

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  using namespace CuEira;

  Configuration configuration(argc, argv);

  FileIO::PlinkReaderFactory plinkReaderFactory;
  FileIO::DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);

  const PersonHandler& personHandler = dataFilesReader->getPersonHandler();
  std::vector<SNP*> snpInfo = dataFilesReader->getSNPInformation(); //FIXME remember to delete
  std::vector<EnvironmentFactor*> environmentFactorInformation = dataFilesReader->getEnvironmentFactorInformation(); //FIXME remember to delete
/*
  const Container::HostVector& outcomes = personHandler.getOutcomes();
  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();

#ifndef CPU
  CuEira::CUDA::LogisticRegressionConfiguration* lrConfig;
#endif

  int numberOfPredictors = 4; //1 for intercept, 1 for snp, 1 for env, 1 for interact
  if(configuration.covariateFileSpecified()){
    const Container::HostMatrix& covariatesMatrix = dataFilesReader->getCovariates();
    numberOfPredictors = +covariatesMatrix.getNumberOfColumns();

#ifndef CPU
    lrConfig = new CuEira::CUDA::LogisticRegressionConfiguration(configuration, hostToDevice, outcomes, kernelWrapper);
    lrConfig->setBetaCoefficents(); //TODO
#endif
  }

#ifdef CPU
  numberOfPredictors--;
  LaVectorDouble * betaCoefficientsLapackpp = new LaVectorDouble(numberOfPredictors + 1);

  const Container::LapackppHostVector& outcomesCast = dynamic_cast<const Container::LapackppHostVector&>(outcomes);
  const LaVectorDouble& outcomesLapackpp = outcomesCast.getLapackpp();

  LaGenMatDouble predictorsLapackpp(numberOfIndividualsToInclude, numberOfPredictors);
  if(configuration.covariateFileSpecified()){
    const Container::HostMatrix& covariatesMatrix = dataFilesReader->getCovariates();

    //Put covariates in matrix
    int numOfCov = covariatesMatrix.getNumberOfColumns();
    for(int row = 0; row < numberOfIndividualsToInclude; ++row){
      for(int col = 0; col < numOfCov; ++col){
        predictorsLapackpp(row, 3 + col) = covariatesMatrix(row, col);
      }
    }
  }
#endif

  //TODO print header

  for(int environmentFactorNumber = 0; environmentFactorNumber < environmentFactorInformation.size();
      ++environmentFactorNumber){
    EnvironmentFactor* environmentFactor = environmentFactorInformation[environmentFactorNumber];
    if(!environmentFactor->getInclude()){
      std::cerr << "Not including environment factor " << environmentFactor->getId().getString() << std::endl;
      continue;
    }

    const Container::HostVector& envHostVector = dataFilesReader->getEnvironmentFactor(*environmentFactor);
    if(!environmentFactor->getInclude()){
      std::cerr << "Environment factor " << environmentFactor->getId().getString() << " got excluded after reading. "
          << std::endl;
      continue;
    }
#ifndef CPU
    lrConfig->setEnvironmentFactor(envHostVector);
#endif

    for(int snpNumber = 0; snpNumber < 1; ++snpNumber){
      SNP* snp = snpInfo[snpNumber];
      if(!snp->getInclude()){
        std::cerr << "Not including SNP " << snp->getId().getString() << std::endl;
        continue;
      }

      Container::SNPVector* snpVector = dataFilesReader->readSNP(*snp);
      if(!snp->getInclude()){
        std::cerr << "SNP " << snp->getId().getString() << " got excluded after reading." << std::endl;
        continue;
      }

      const Container::HostVector* snpVectorModData = snpVector->getRecodedData();

#ifdef CPU

      //fill predictorsLapackpp
      for(int row = 0; row < numberOfIndividualsToInclude; ++row){
        //SNP
        predictorsLapackpp(row, 0) = (*snpVectorModData)(row);

        //ENV
        predictorsLapackpp(row, 1) = envHostVector(row);

        //Interaction
        predictorsLapackpp(row, 2) = (*snpVectorModData)(row) * envHostVector(row);
      }

      //Reset beta
      for(int i = 0; i < betaCoefficientsLapackpp->rows(); ++i){
        (*betaCoefficientsLapackpp)(i) = 0;
      }

      LogisticRegression::MultipleLogisticRegression logisticRegression(predictorsLapackpp, outcomesLapackpp,
          betaCoefficientsLapackpp);
      logisticRegression.calculate();

      const LaVectorDouble& beta = logisticRegression.getBeta();
#else
      lrConfig->setSNP();
      lrConfig->setInteraction();

      LogisticRegression lr(lrConfig);
      const Container::DeviceVector& deviceBeta=lr.getBeta();
      deviceToHost.transferVector(deviceBeta); //TODO
#endif

      double intercept = beta(0);
      double snpBeta = beta(1);
      double envBeta = beta(2);
      double interactionBeta = beta(3);

      Recode recode = ALL_RISK;
      if(snpBeta < 0 && snpBeta < envBeta && snpBeta < interactionBeta){
        recode = SNP_PROTECT;
      }else if(envBeta < 0 && envBeta < snpBeta && envBeta < interactionBeta){
        recode = ENVIRONMENT_PROTECT;
      }else if(interactionBeta < 0 && interactionBeta < snpBeta && interactionBeta < envBeta){
        recode = INTERACTION_PROTECT;
      }

      //Change stuff based on recode
      if(recode == 1 || recode == 3){
        snpVector->recode(recode);
      }

      if(recode == 2 || recode == 3){

      }

      //Recalculate if needed
      //TODO
      if(recode != ALL_RISK){

      }

      //TODO do statistics

    } //for snp
  } //for environment

  //for(int snpNumber = 0; snpNumber < snpInfo.size(); ++snpNumber){
  for(int snpNumber = 0; snpNumber < 1; ++snpNumber){

  } //for snp

  delete dataFilesReader;
  */
}
