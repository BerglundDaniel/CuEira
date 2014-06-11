#include <ConstructorHelpers.h>

namespace CuEira {
namespace CuEira_Test {

ConstructorHelpers::ConstructorHelpers() {
  srand(time(NULL));
}

ConstructorHelpers::~ConstructorHelpers() {

}

Person* ConstructorHelpers::constructPersonInclude(int number) {
  std::ostringstream os;
  os << "ind" << number;
  Id id(os.str());
  Sex sex;
  Phenotype phenotype;

  if(rand() % 2 == 0){
    sex = MALE;
  }else{
    sex = FEMALE;
  }

  if(rand() % 2 == 0){
    phenotype = AFFECTED;
  }else{
    phenotype = UNAFFECTED;
  }

  return new Person(id, sex, phenotype, true);
}

Person* ConstructorHelpers::constructPersonNotInclude(int number) {
  std::ostringstream os;
  os << "ind" << number;
  Id id(os.str());
  Sex sex;

  if(rand() % 2 == 0){
    sex = MALE;
  }else{
    sex = FEMALE;
  }

  return new Person(id, sex, MISSING, false);
}

Person* ConstructorHelpers::constructPersonInclude(int number, Phenotype phenotype) {
  std::ostringstream os;
  os << "ind" << number;
  Id id(os.str());
  Sex sex;

  if(rand() % 2 == 0){
    sex = MALE;
  }else{
    sex = FEMALE;
  }

  return new Person(id, sex, phenotype, true);
}

Container::EnvironmentVectorMock* ConstructorHelpers::constructEnvironmentVectorMock() {
  const int numberOfIndividuals = 3;
  const int numberOfColumns = 2;

#ifdef CPU
  Container::HostMatrix* dataMatrix= new Container::LapackppHostMatrix(
      new LaGenMatDouble(numberOfIndividuals, numberOfColumns));
#else
  Container::HostMatrix* dataMatrix = new Container::PinnedHostMatrix(numberOfIndividuals, numberOfColumns);
#endif

  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfColumns);
  for(int i = 0; i < numberOfColumns; ++i){
    std::ostringstream os;
    os << "envfactor" << i;
    Id id(os.str());
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

  EnvironmentFactorHandler environmentHandler(dataMatrix, environmentFactors);
  return new Container::EnvironmentVectorMock(environmentHandler, *(*environmentFactors)[0]);
}

Container::SNPVectorMock* ConstructorHelpers::constructSNPVectorMock() {
  const int numberOfData = 2;

  SNP snp(Id("test_snp1"), "allele1", "allele2", 1);
  snp.setAllAlleleFrequencies(1, 1);
  snp.setCaseAlleleFrequencies(1, 1);
  snp.setControlAlleleFrequencies(1, 1);
  snp.setMinorAlleleFrequency(1);
  snp.setRiskAllele(ALLELE_ONE);

  std::vector<int>* SNPData = new std::vector<int>(numberOfData);
  for(int i = 0; i < numberOfData; ++i){
    (*SNPData)[i] = 0;
  }

  return new Container::SNPVectorMock(SNPData, snp);
}

} /* namespace CuEira_Test */
} /* namespace CuEira */
