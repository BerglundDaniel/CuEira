#include "RegularHostMatrix.h"

namespace CuEira {
namespace Container {

RegularHostMatrix::RegularHostMatrix(int numberOfRows, int numberOfColumns) :
    HostMatrix(ceil(((double) numberOfRows) / CPU_UNROLL) * CPU_UNROLL, numberOfColumns, numberOfRows, numberOfColumns,
        nullptr) {
  hostMatrix = (PRECISION*) malloc(sizeof(PRECISION) * numberOfRealRows * numberOfRealColumns);
}

RegularHostMatrix::~RegularHostMatrix() {
  free(hostMatrix);
}

RegularHostVector* RegularHostMatrix::operator()(int column) {
#ifdef DEBUG
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new RegularHostVector(numberOfRows, hostVector, true);
}

const RegularHostVector* RegularHostMatrix::operator()(int column) const {
#ifdef DEBUG
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  PRECISION* hostVector = hostMatrix + numberOfRows * column;
  return new RegularHostVector(numberOfRows, hostVector, true);
}

PRECISION& RegularHostMatrix::operator()(int row, int column) {
#ifdef DEBUG
  if(row >= numberOfRows || row < 0){
    std::ostringstream os;
    os << "Index " << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  return *(hostMatrix + (numberOfRealRows * column) + row);
}

const PRECISION& RegularHostMatrix::operator()(int row, int column) const {
#ifdef DEBUG
  if(row >= numberOfRows || row < 0){
    std::ostringstream os;
    os << "Index " << row << " is larger than the number of rows " << numberOfRows << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
  if(column >= numberOfColumns || column < 0){
    std::ostringstream os;
    os << "Index " << column << " is larger than the number of columns " << numberOfColumns << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  return *(hostMatrix + (numberOfRealRows * column) + row);
}

}
/* namespace Container */
} /* namespace CuEira */
