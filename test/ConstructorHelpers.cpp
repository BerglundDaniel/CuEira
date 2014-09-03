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
  return new Container::EnvironmentVectorMock();
}

Container::SNPVectorMock* ConstructorHelpers::constructSNPVectorMock() {
  SNP snp(Id("test_snp1"), "allele1", "allele2", 1);
  snp.setRiskAllele(ALLELE_ONE);
  return new Container::SNPVectorMock(snp);
}

EnvironmentFactorHandlerMock* ConstructorHelpers::constructEnvironmentFactorHandlerMock() {
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

  return new EnvironmentFactorHandlerMock(dataMatrix, environmentFactors);
}

FileIO::BedReaderMock* ConstructorHelpers::constructBedReaderMock() {
  ConfigurationMock configurationMock;
  EXPECT_CALL(configurationMock, getGeneticModel()).WillRepeatedly(Return(DOMINANT));
  EXPECT_CALL(configurationMock, getMinorAlleleFrequencyThreshold()).WillRepeatedly(Return(0.05));

  return new FileIO::BedReaderMock(configurationMock, Container::SNPVectorFactoryMock(configurationMock),
      PersonHandlerMock());
}

Container::SNPVectorFactoryMock* ConstructorHelpers::constructSNPVectorFactoryMock() {
  ConfigurationMock configurationMock;
  EXPECT_CALL(configurationMock, getGeneticModel()).WillRepeatedly(Return(DOMINANT));
  EXPECT_CALL(configurationMock, getMinorAlleleFrequencyThreshold()).WillRepeatedly(Return(0.05));

  return new Container::SNPVectorFactoryMock(configurationMock);
}

ContingencyTableFactoryMock ConstructorHelpers::constructContingencyTableFactoryMock() {
  const int size = 3;
#ifdef CPU
  Container::LapackppHostVector outcomes(new LaVectorDouble(size));
#else
  Container::PinnedHostVector outcomes(size);
#endif
  return new ContingencyTableFactoryMock(outcomes);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
