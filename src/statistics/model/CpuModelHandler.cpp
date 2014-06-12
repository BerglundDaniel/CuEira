#include "CpuModelHandler.h"

namespace CuEira {
namespace Model {

CpuModelHandler::CpuModelHandler(DataHandler& dataHandler, Container::HostMatrix* covariates) :
    ModelHandler(dataHandler), covariates(covariates) {

}

CpuModelHandler::~CpuModelHandler() {
  delete covariates;
}

Statistics* CpuModelHandler::calculateModel() {
  //TODO calc modlel

  const Container::HostVector& betaCoefficents; //FIXME
  const Container::HostVector& standardError; //FIXME

  //Does any of the data need to be recoded?
  Recode recode;
  double snpBeta = betaCoefficents(1);
  double envBeta = betaCoefficents(2);
  double interactionBeta = betaCoefficents(3);

  if(snpBeta < 0 && snpBeta < envBeta && snpBeta < interactionBeta){
    recode = SNP_PROTECT;
  }else if(envBeta < 0 && envBeta < snpBeta && envBeta < interactionBeta){
    recode = ENVIRONMENT_PROTECT;
  }else if(interactionBeta < 0 && interactionBeta < snpBeta && interactionBeta < envBeta){
    recode = INTERACTION_PROTECT;
  }

  if(recode != ALL_RISK){
    dataHandler.recode(recode);

    //Calculate again
  }

  return new Statistics(betaCoefficents, standardError);
}

} /* namespace Model */
} /* namespace CuEira */
