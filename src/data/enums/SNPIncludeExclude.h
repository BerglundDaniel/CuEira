#ifndef SNPINCLUDEEXCLUDE_H_H
#define SNPINCLUDEEXCLUDE_H_H

namespace CuEira {

/**
 * This enum represents different reasons for why a SNP has been excluded from the analysis
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
enum SNPIncludeExclude {
  INCLUDE, LOW_MAF, LOW_CELL_NUMBER, NEGATIVE_POSITION
};

} /* namespace CuEira */

#endif /* SNPINCLUDEEXCLUDE_H_H */
