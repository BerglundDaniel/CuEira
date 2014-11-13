#include "DataHandler.h"

namespace CuEira {

#ifdef PROFILE
  boost::chrono::duration<long long, boost::nano> DataHandler::timeSpentRecode;
  boost::chrono::duration<long long, boost::nano> timeSpentNext;
  boost::chrono::duration<long long, boost::nano> timeSpentSNPRead;
  boost::chrono::duration<long long, boost::nano> timeSpentStatModel;
  std::mutex mutex;
  bool firstDestroy = true;
#endif

DataHandler::DataHandler(const Configuration& configuration, FileIO::BedReader& bedReader,
    const ContingencyTableFactory& contingencyTableFactory,
    const Model::ModelInformationFactory& modelInformationFactory,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(&dataQueue), bedReader(&bedReader), interactionVector(
        interactionVector), snpVector(nullptr), environmentVector(environmentVector), environmentInformation(
        &environmentInformation), currentEnvironmentFactorPos(environmentInformation.size() - 1), state(
        NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(&contingencyTableFactory), modelInformationFactory(
        &modelInformationFactory), currentSNP(nullptr), cellCountThreshold(configuration.getCellCountThreshold()), alleleStatistics(
        nullptr), modelInformation(nullptr), currentEnvironmentFactor(nullptr), currentStatisticModel(ADDITIVE), appliedStatisticModel(
        false) {

}

DataHandler::DataHandler(const Configuration& configuration) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(nullptr), bedReader(nullptr), interactionVector(
        nullptr), snpVector(nullptr), environmentVector(nullptr), environmentInformation(nullptr), currentEnvironmentFactorPos(
        0), state(NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(nullptr), currentSNP(nullptr), alleleStatistics(
        nullptr), cellCountThreshold(0), modelInformationFactory(nullptr), modelInformation(nullptr), currentEnvironmentFactor(
        nullptr), currentStatisticModel(ADDITIVE), appliedStatisticModel(false) {

}

DataHandler::~DataHandler() {
#ifdef PROFILE
  mutex.lock();

  if(firstDestroy){
    firstDestroy = false;
    mutex.unlock():
    std::cerr << "DataHandler, time spent recode: " << boost::chrono::duration_cast<boost::chrono::millioseconds>(timeSpentRecode) << std::endl;
    std::cerr << "DataHandler, time spent next: " << boost::chrono::duration_cast<boost::chrono::millioseconds>(timeSpentNext) << std::endl;
    std::cerr << "DataHandler, time spent read snp: " << boost::chrono::duration_cast<boost::chrono::millioseconds>(timeSpentSNPRead) << std::endl;
    std::cerr << "DataHandler, time spent statistic model: " << boost::chrono::duration_cast<boost::chrono::millioseconds>(timeSpentStatModel) << std::endl;
  }else{
    mutex.unlock():
  }

#endif

  delete interactionVector;
  delete snpVector;
  delete environmentVector;
  delete contingencyTable;
  delete alleleStatistics;
  delete modelInformation;
  delete currentSNP;
}

DataHandlerState DataHandler::next() {
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

  currentRecode = ALL_RISK;
  appliedStatisticModel = false;

  delete contingencyTable;
  delete modelInformation;

  contingencyTable = nullptr;
  modelInformation = nullptr;
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED;
  }
#endif

  if(currentEnvironmentFactorPos == environmentInformation->size() - 1){ //Check if we were at the last EnvironmentFactor so should start with next snp
    delete currentSNP;
    delete snpVector;
    delete alleleStatistics;
    currentSNP = nullptr;
    snpVector = nullptr;
    alleleStatistics = nullptr;

    currentSNP = dataQueue->next();
    if(currentSNP == nullptr){
#ifdef PROFILE
      boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
      timeSpentNext+=after - before;
#endif

      return DONE;
    }

    bool include = readSNP(*currentSNP);
    if(!include){
      modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *(*environmentInformation)[0],
          *alleleStatistics);
#ifdef PROFILE
      boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
      timeSpentNext+=after - before;
#endif

      return SKIP;
    }
    currentEnvironmentFactorPos = 0;
  }else{
    currentEnvironmentFactorPos++;
    snpVector->recode(ALL_RISK);
  }

  currentEnvironmentFactor = (*environmentInformation)[currentEnvironmentFactorPos];
  environmentVector->switchEnvironmentFactor(*currentEnvironmentFactor);
  interactionVector->recode(*snpVector);

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

    setSNPInclude(*currentSNP, *contingencyTable);
    if(!currentSNP->shouldInclude()){
      modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
          *alleleStatistics, *contingencyTable);

#ifdef PROFILE
      boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
      timeSpentNext+=after - before;
#endif

      return SKIP;
    }
  }

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics, *contingencyTable);
  }else{
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics);
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentNext+=after - before;
#endif

  return CALCULATE;
}

void DataHandler::applyStatisticModel(StatisticModel statisticModel) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using applyStatisticModel run next() at least once.");
  }
#endif

  if(appliedStatisticModel){ // If a model has been previously applied after a next or a recode then things can have changed
    snpVector->recode(currentRecode);
    environmentVector->recode(currentRecode);
  }

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());

  appliedStatisticModel = true;
  currentStatisticModel = statisticModel;

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentStatModel+=after - before;
#endif
}

void DataHandler::recode(Recode recode) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using recode run next() at least once.");
  }
#endif

  if(recode == currentRecode){
#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentRecode+=after - before;
#endif

    return;
  }
#ifdef DEBUG
  else if(!(recode == SNP_PROTECT || recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT || recode == ALL_RISK)){
    throw InvalidState("Unknown recode for DataHandler.");
  }
#endif
  currentRecode = recode;
  appliedStatisticModel = false;

  snpVector->recode(recode);
  environmentVector->recode(recode);
  interactionVector->recode(*snpVector);

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    delete contingencyTable;
    delete modelInformation;
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics, *contingencyTable);
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentRecode+=after - before;
#endif
}

bool DataHandler::readSNP(SNP& nextSnp) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeRead = boost::chrono::system_clock::now();
#endif

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader->readSNP(nextSnp);

#ifdef PROFILE
  boost::chrono::system_clock::time_point afterRead = boost::chrono::system_clock::now();
  timeSpentSNPRead+=afterRead - beforeRead;
#endif

  alleleStatistics = pair->first;
  snpVector = pair->second;

  delete pair;
  return nextSnp.shouldInclude();
}

Recode DataHandler::getRecode() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return currentRecode;
}

const Model::ModelInformation& DataHandler::getCurrentModelInformation() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getModelInformation run next() at least once.");
  }
#endif
  return *modelInformation;
}

const SNP& DataHandler::getCurrentSNP() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentSNP use next() at least once.");
  }
#endif
  return *currentSNP;
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentEnvironmentFactor use next() at least once.");
  }
#endif
  return *(*environmentInformation)[currentEnvironmentFactorPos];
}

const Container::SNPVector& DataHandler::getSNPVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getSNPVector run next() at least once.");
  }
#endif
  return *snpVector;
}

const Container::InteractionVector& DataHandler::getInteractionVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getInteractionVector run next() at least once.");
  }
#endif
  return *interactionVector;
}

const Container::EnvironmentVector& DataHandler::getEnvironmentVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getEnvironmentVector run next() at least once.");
  }
#endif
  return *environmentVector;
}

void DataHandler::setSNPInclude(SNP& snp, const ContingencyTable& contingencyTable) const {
  const std::vector<int>& table = contingencyTable.getTable();
  const int size = table.size();
  for(int i = 0; i < size; ++i){
    if(table[i] <= cellCountThreshold){
      snp.setInclude(LOW_CELL_NUMBER);
      break;
    }
  }
}

} /* namespace CuEira */
