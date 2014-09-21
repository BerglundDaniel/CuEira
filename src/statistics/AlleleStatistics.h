#ifndef ALLELESTATISTICS_H_
#define ALLELESTATISTICS_H_

#define ALLELE_ONE_CASE_POSITION 0
#define ALLELE_TWO_CASE_POSITION 1
#define ALLELE_ONE_CONTROL_POSITION 2
#define ALLELE_TWO_CONTROL_POSITION 3
#define ALLELE_ONE_ALL_POSITION 4
#define ALLELE_TWO_ALL_POSITION 5

#include <vector>
#include <ostream>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class AlleleStatistics {
  friend std::ostream& operator<<(std::ostream& os, const AlleleStatistics& alleleStatistics);
public:
  AlleleStatistics(const std::vector<int>* numberOfAlleles, const std::vector<double>* alleleFrequencies);
  virtual ~AlleleStatistics();

  virtual const std::vector<int>& getAlleleNumbers() const;
  virtual const std::vector<double>& getAlleleFrequencies() const;

  AlleleStatistics(const AlleleStatistics&) = delete;
  AlleleStatistics(AlleleStatistics&&) = delete;
  AlleleStatistics& operator=(const AlleleStatistics&) = delete;
  AlleleStatistics& operator=(AlleleStatistics&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const;

private:
  const std::vector<int>* numberOfAlleles;
  const std::vector<double>* alleleFrequencies;
};

} /* namespace CuEira */

#endif /* ALLELESTATISTICS_H_ */
