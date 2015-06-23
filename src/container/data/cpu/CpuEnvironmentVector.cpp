#include "CpuEnvironmentVector.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuEnvironmentVector::CpuEnvironmentVector(const EnvironmentFactorHandler<RegularHostVector>& environmentFactorHandler) :
    EnvironmentVector(environmentFactorHandler){

}

CpuEnvironmentVector::~CpuEnvironmentVector(){

}

void CpuEnvironmentVector::recodeProtective(){
  int c;
  if(environmentFactor.getVariableType() == BINARY){
    c = 1;
  }else{
    c = environmentFactor.getMax() + environmentFactor.getMin();
  }

  //UNROLL
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*envExMissing)(i) = c - (*envExMissing)(i);
  }
}

void CpuEnvironmentVector::recodeAllRisk(){
  Blas::copyVector(originalData, *envExMissing);
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
