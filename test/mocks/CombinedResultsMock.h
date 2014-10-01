#ifndef COMBINEDRESULTSMOCK_H_
#define COMBINEDRESULTSMOCK_H_

#include <gmock/gmock.h>
#include <ostream>

#include <CombinedResults.h>
#include <Recode.h>

namespace CuEira {
namespace Model {

class CombinedResultsMock: public CombinedResults {
public:
  CombinedResultsMock() :
  CombinedResults(){

  }

  virtual ~CombinedResultsMock() {

  }

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));

};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTSMOCK_H_ */
