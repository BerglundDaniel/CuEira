#include <iostream>
#include <stdexcept>

#include <Configuration.h>
//#include <BimReader.h>
//#include <BedReader.h>

/**
 * This is the main part
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  CuEira::Configuration configuration = CuEira::Configuration(argc, argv);
  //CuEira::FileIO::BimReader bimReader(configuration);
  //CuEira::FileIO::BedReader bedReader(configuration);
}
