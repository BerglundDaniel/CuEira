#include <iostream>
#include <stdexcept>

#include <Configuration.h>

/**
 * This is the main part
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  std::cerr << "argc " << argc << std::endl;
  std::cerr << "argv " << *argv << std::endl;

  CuEira::Configuration configuration(argc, argv);

  /*
  PlinkReaderFactory plinkReaderFactory();
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);
  std::vector<SNP*> snpInfo = dataFilesReader->getSNPInfo();
  std::vector<EnvironmentFactor> environmentFactorInfo = dataFilesReader->getEnvironmentFactorInfo();

  int numberOfIndividualsToInclude = dataFilesReader->getNumberOfIndividualsToInclude();
  int numberOfPreds = 3; //1 for snp, 1 for env, 1 for interact
  //HostMatrix covariatesMatrix = dataFilesReader->getCovariates(); TODO
  LaVectorDouble * betaCoefficientsLapackpp = new LaVectorDouble(numberOfPreds + 1);

  //TODO convert outcomes to lappackpphost
  HostVector outcomes = dataFilesReader->getOutcomes();
  LaVectorDouble& outcomesLapackpp = outcomes.getLapackpp();

  for(int environmentFactorNumber = 0; environmentFactorNumber < environmentFactorInfo.size();
      ++environmentFactorNumber){
    EnivironmentFactor enivironmentFactor = environmentFactorInfo[environmentFactorNumber];
    HostVector environmentHostVector = DataFilesReader->getEnvironmentFactor(environmentFactor);

    for(int snpNumber = 0; snpNumber < snpInfo.size(); ++snpNumber){
      SNP snp = snpInfo[snpNumber];
      HostVector* snpVector = dataFilesReader->readSNP(snp);
      LaGenMatDouble predictorsLapackpp(numberOfIndividualsToInclude, numberOfPreds);

      //fill predictorsLapackpp
      for(int row = 0; row < numberOfIndividualsToInclude; ++row){
        //SNP
        predictorsLapackpp(row, 0) = snpVector(row);

        //ENV
        predictorsLapackpp(row, 1) = environmentHostVector(row);

        //Interaction
        predictorsLapackpp(row, 2) = snpVector(row) * environmentHostVector(row);
      }

      //Reset beta
      for(int i = 0; i < betaCoefficientsLapackpp.size(); ++i){
        (*betaCoefficientsLapackpp)(i) = 0;
      }

      MultipleLogisticRegression logisticRegression(predictorsLapackpp, outcomesLapackpp, betaCoefficientsLapackpp);
      logisticRegression.calculate();

      //Get stuff
      //const LaGenMatDouble& getInformationMatrix();
      //double getLogLikelihood();
      //betaCoefficientsLapackpp

      //Recode?
      double snpBeta = (*betaCoefficientsLapackpp)(0);
      double envBeta = (*betaCoefficientsLapackpp)(1);
      double interactionBeta = (*betaCoefficientsLapackpp)(2);
      int recode = 0;

      if(){
        recode = 1;
      }else if(){
        recode = 2;
      }else if(){
        receode = 3;
      }

      double cA0B1 = coef.getEntry(MATRIX_INDEX_A0B1); //env
      double cA1B0 = coef.getEntry(MATRIX_INDEX_A1B0); //snp
      double cA1B1 = coef.getEntry(MATRIX_INDEX_A1B1); //interaction

      // Recalculate the risk alleles if necessary. The recode values are
      // described below.
      if(cA1B0 < 0 && cA1B0 < cA0B1 && cA1B0 < cA1B1){
        recode = 1;
      }else if(cA0B1 < 0 && cA0B1 < cA1B0 && cA0B1 < cA1B1){
        recode = 2;
      }else if(cA1B1 < 0 && cA1B1 < cA1B0 && cA1B1 < cA0B1){
        recode = 3;
      }

      //Change stuff based on recode
      if(recode == 1 || recode == 3){

      }

      if(recode == 2 || recode == 3){

      }

      //Recalculate if needed
      if(recode>0){

      }

      //Do statistics

      delete snpVector;
    } //for snp
  } // for env
  */
}
