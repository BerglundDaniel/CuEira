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
 * This is the main part
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  using namespace CuEira;

  Configuration configuration(argc, argv);

  FileIO::PlinkReaderFactory plinkReaderFactory;
  FileIO::DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);

#ifdef CPU
  const PersonHandler& personHandler = dataFilesReader->getPersonHandler();
  std::vector<SNP*> snpInfo = dataFilesReader->getSNPInformation();
  std::vector<EnvironmentFactor*> environmentFactorInformation = dataFilesReader->getEnvironmentFactorInformation();

  const int numberOfIndividualsToInclude = personHandler.getNumberOfIndividualsToInclude();
  const int numberOfPreds = 3; //1 for snp, 1 for env, 1 for interact
  LaVectorDouble * betaCoefficientsLapackpp = new LaVectorDouble(numberOfPreds + 1);

  //HostMatrix covariatesMatrix = dataFilesReader->getCovariates(); //Skip for now TODO

  const Container::HostVector& outcomes = personHandler.getOutcomes();
  const Container::LapackppHostVector& outcomesCast = dynamic_cast<const Container::LapackppHostVector&>(outcomes);
  const LaVectorDouble& outcomesLapackpp = outcomesCast.getLapackpp();

  EnvironmentFactor* environmentFactor = environmentFactorInformation[1];//Just using one atm
  const Container::HostVector& envHostVector = dataFilesReader->getEnvironmentFactor(*environmentFactor);

  //std::cout << "ID intercept snp env interaction" << std::endl;
  /*
   std::cout << "Snp alleone alleletwo riskallele allfreq1 allfreq2 casefreq1 casefreq2 controlfreq1 controlfreq2"
   << std::endl;
   */

  std::cout << "outcome snp env interaction" << std::endl;
  std::cerr << "invidiauls: " << numberOfIndividualsToInclude << std::endl;

  //for(int snpNumber = 0; snpNumber < snpInfo.size(); ++snpNumber){
  for(int snpNumber = 0; snpNumber < 1; ++snpNumber){
    SNP* snp = snpInfo[snpNumber];
    if(!snp->getInclude()){
      std::cerr << "Not including SNP " << snp->getId().getString() << std::endl;
      continue;
    }

    Container::SNPVector* snpVector = dataFilesReader->readSNP(*snp);
    if(!snp->getInclude()){
      std::cerr << "SNP " << snp->getId().getString() << " got exluded after reading." << std::endl;
      continue;
    }

    LaGenMatDouble predictorsLapackpp(numberOfIndividualsToInclude, numberOfPreds);
    const Container::HostVector* snpVectorModData = snpVector->getRecodedData();

    //fill predictorsLapackpp
    for(int row = 0; row < numberOfIndividualsToInclude; ++row){
      //SNP
      predictorsLapackpp(row, 0) = (*snpVectorModData)(row);

      //ENV
      predictorsLapackpp(row, 1) = envHostVector(row);

      //Interaction
      predictorsLapackpp(row, 2) = (*snpVectorModData)(row) * envHostVector(row);

      std::cout << outcomesLapackpp(row) << " " << (*snpVectorModData)(row) << " " << envHostVector(row) << " "
      << (*snpVectorModData)(row) * envHostVector(row) << std::endl;
    }

    //Reset beta
    for(int i = 0; i < betaCoefficientsLapackpp->rows(); ++i){
      (*betaCoefficientsLapackpp)(i) = 0;
    }

    LogisticRegression::MultipleLogisticRegression logisticRegression(predictorsLapackpp, outcomesLapackpp,
        betaCoefficientsLapackpp);
    logisticRegression.calculate();

    const LaVectorDouble& beta = logisticRegression.getBeta();

    double intercept = beta(0);
    double snpBeta = beta(1);
    double envBeta = beta(2);
    double interactionBeta = beta(3);

    //Recode?
    /*
     double intercept = (*betaCoefficientsLapackpp)(0); //env
     double snpBeta = (*betaCoefficientsLapackpp)(1); //interact
     double envBeta = (*betaCoefficientsLapackpp)(2); //snp
     double interactionBeta = (*betaCoefficientsLapackpp)(3); //intercept
     */

    /*int recode = 0;

     if(snpBeta < 0 && snpBeta < envBeta && snpBeta < interactionBeta){
     recode = 1;
     }else if(envBeta < 0 && envBeta < snpBeta && envBeta < interactionBeta){
     recode = 2;
     }else if(interactionBeta < 0 && interactionBeta < snpBeta && interactionBeta < envBeta){
     recode = 3;
     }

     std::cerr << "recode " << recode << std::endl;*/
    //std::cout << snpNumber << " " << intercept << " " << snpBeta << " " << envBeta << " " << interactionBeta
    //<< std::endl;
    /*std::cout << snpNumber << " " << snp->getAlleleOneName() << " " << snp->getAlleleTwoName() << " "
     << snp->getRiskAllele() << " " << snp->getAlleleOneAllFrequency() << " " << snp->getAlleleTwoAllFrequency()
     << " " << snp->getAlleleOneCaseFrequency() << " " << snp->getAlleleTwoCaseFrequency() << " "
     << snp->getAlleleOneControlFrequency() << " " << snp->getAlleleTwoControlFrequency() << std::endl;*/

    //if(interactionBeta > (snpBeta + envBeta - 1)){
    //std::cout << "interaction" << std::endl;
    //}
    /*
     //Change stuff based on recode
     if(recode == 1 || recode == 3){

     }

     if(recode == 2 || recode == 3){

     }

     //Recalculate if needed
     if(recode > 0){

     }

     //Do statistics
     */
    delete snpVector;
  } //for snp
#endif

  delete dataFilesReader;
}
