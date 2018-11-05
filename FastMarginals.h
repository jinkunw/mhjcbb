#pragma once
#include <boost/unordered_map.hpp>
#include <unordered_map>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace gtsam {

/*
 * Implementation of the covariance recovery algorithm in
 *     Covariance Recovery from a Square Root Information Matrix for Data Association
 * See code in iSAM for details
 *     http://people.csail.mit.edu/kaess/isam/
 */
struct SparseBlockVector {
  std::map<Key, Matrix> vector;

  void insert(Key i, const Matrix &block) {
    vector[i] = block;
  }

  std::map<Key, Matrix>::const_iterator begin() const { return vector.begin(); };
  std::map<Key, Matrix>::const_iterator end() const { return vector.end(); };

  SparseBlockVector() {}
};

std::ostream &operator<<(std::ostream &os, const SparseBlockVector &vector);

typedef std::pair<Key, Key> KeyPair;
std::size_t hash_value(const KeyPair &key_pair);

struct CovarianceCache {
  boost::unordered_map<KeyPair, Matrix> entries;
  std::unordered_map<Key, Matrix> diag;
  std::unordered_map<Key, SparseBlockVector> rows;

  CovarianceCache() {}
};

class FastMarginals {
 public:
  FastMarginals(const ISAM2 &isam2) : isam2_(isam2) {
    initialize();
  }

  Matrix marginalCovariance(const Key &variable);

  Matrix jointMarginalCovariance(const KeyVector &variables);

 protected:
  void initialize();

  Matrix getRBlock(const Key &key_i, const Key &key_j);

  const SparseBlockVector &getRRow(const Key &key);

  Matrix getR(const std::vector<Key> &variables);

  Matrix getKeyDiag(const Key &key);

  size_t getKeyDim(const Key &key);

  Matrix sumJ(const Key key_l, const Key key_i);

  Matrix recover(const Key &key_i, const Key &key_l);

  const ISAM2 &isam2_;
  std::vector<Key> ordering_;
  boost::unordered_map<Key, size_t> key_idx_;
  CovarianceCache cov_cache_;
  std::unordered_map<Key, Matrix> Fs_;
  std::unordered_map<Key, Matrix> F_;
  Key last_key_;
  size_t size0_;
};

}  // namespace gtsam
