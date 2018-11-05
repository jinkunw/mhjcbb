#include <gtsam/nonlinear/FastMarginals.h>
#include <queue>

namespace gtsam {
std::ostream &operator<<(std::ostream &os, const SparseBlockVector &vector) {
  for (auto it = vector.begin(); it != vector.end(); ++it)
    os << it->first << std::endl
       << it->second << std::endl;
  return os;
}

std::size_t hash_value(const KeyPair &key_pair) {
  std::size_t seed = 0;
  boost::hash_combine(seed, key_pair.first);
  boost::hash_combine(seed, key_pair.second);
  return seed;
}

Matrix FastMarginals::marginalCovariance(const Key &variable) {
  return recover(variable, variable);
}

Matrix FastMarginals::jointMarginalCovariance(const KeyVector &variables) {
  size_t dim = 0;
  std::vector<size_t> variable_acc_dim;
  for (Key key : variables) {
    variable_acc_dim.push_back(dim);
    dim += getKeyDim(key);
  }

  std::vector<int> variable_idx(variables.size());
  std::iota(variable_idx.begin(), variable_idx.end(), 0);
  std::sort(variable_idx.begin(), variable_idx.end(),
            [this, &variables](int i, int j) {
              return key_idx_[variables[i]] < key_idx_[variables[j]];
            });

  Matrix cov = Matrix::Zero(dim, dim);
  for (int j = variable_idx.size() - 1; j >= 0; --j) {
    Key key_j = variables[variable_idx[j]];
    size_t col = variable_acc_dim[variable_idx[j]];

    for (int i = j; i >= 0; --i) {
      Key key_i = variables[variable_idx[i]];
      size_t row = variable_acc_dim[variable_idx[i]];

      if (row > col) {
        cov.block(col, row, getKeyDim(key_j), getKeyDim(key_i)) = recover(key_i, key_j).transpose();
      } else {
        cov.block(row, col, getKeyDim(key_i), getKeyDim(key_j)) = recover(key_i, key_j);
      }
    }
  }
  cov.triangularView<Eigen::Lower>() = cov.transpose();
  return cov;
}

Matrix FastMarginals::getRBlock(const Key &key_i, const Key &key_j) {
  ISAM2Clique::shared_ptr clique = isam2_[key_i];
  const ISAM2Clique::sharedConditional conditional = clique->conditional();
  if (conditional->find(key_j) == conditional->end())
    return Matrix();

  size_t block_row = conditional->find(key_i) - conditional->begin();
  size_t block_col = conditional->find(key_j) - conditional->begin();
  const auto &m = conditional->matrixObject();
  DenseIndex row = m.offset(block_row);
  DenseIndex col = m.offset(block_col);
  return m.matrix().block(row, col, getKeyDim(key_i), getKeyDim(key_j));
}

const SparseBlockVector &FastMarginals::getRRow(const Key &key) {
  const auto &it = cov_cache_.rows.find(key);
  if (it == cov_cache_.rows.cend()) {
    auto ret = cov_cache_.rows.insert(std::make_pair(key, SparseBlockVector()));
    bool started = false;
    for (Key key_i : ordering_) {
      if (key_i == key)
        started = true;
      if (!started)
        continue;

      Matrix block = getRBlock(key, key_i);
      if (block.size() > 0)
        ret.first->second.insert(key_i, block);
    }
    return ret.first->second;
  } else
    return it->second;
}

Matrix FastMarginals::getR(const std::vector<Key> &variables) {
  size_t dim = 0;
  for (Key key : variables)
    dim += getKeyDim(key);

  Matrix R = Matrix::Zero(dim, dim);
  size_t row = 0;
  for (size_t i = 0; i < variables.size(); ++i) {
    Key key_i = variables[i];
    size_t col = row;
    size_t dim_i = getKeyDim(key_i);
    for (size_t j = i; j < variables.size(); ++j) {
      Key key_j = variables[j];
      size_t dim_j = getKeyDim(key_j);
      Matrix block = getRBlock(key_i, key_j);
      if (block.size() > 0)
        R.block(row, col, dim_i, dim_j) = block;
      col += dim_j;
    }
    row += dim_i;
  }
  return R;
}

size_t FastMarginals::getKeyDim(const Key &key) {
  return isam2_.getLinearizationPoint().at(key).dim();
}

Matrix FastMarginals::getKeyDiag(const Key &key) {
  auto it = cov_cache_.diag.find(key);
  if (it == cov_cache_.diag.end()) {
    auto ret = cov_cache_.diag.insert(std::make_pair(key, getRBlock(key, key).inverse()));
    return ret.first->second;
  } else
    return it->second;
}

void FastMarginals::initialize() {
  std::queue<ISAM2Clique::shared_ptr> q;
  assert(isam2_.roots().size() == 1);
  q.push(isam2_.roots()[0]);
  while (!q.empty()) {
    ISAM2Clique::shared_ptr c = q.front();
    q.pop();
    std::vector<Key> sub;
    assert(c->conditional() != nullptr);
    for (Key key : c->conditional()->frontals()) {
      sub.push_back(key);
    }
    ordering_.insert(ordering_.begin(), sub.begin(), sub.end());
    for (auto child : c->children)
      q.push(child);
  }

  for (size_t i = 0; i < ordering_.size(); ++i) {
    Key key = ordering_[i];
    key_idx_[key] = i;
  }
}

Matrix FastMarginals::sumJ(const Key key_l, const Key key_i) {
  Matrix sum = Matrix::Zero(getKeyDim(key_i), getKeyDim(key_l));
  const SparseBlockVector &Ri = getRRow(key_i);

  size_t idx_l = key_idx_[key_l];
  size_t idx_i = key_idx_[key_i];
  for (auto it = Ri.begin(); it != Ri.end(); ++it) {
    Key key_j = it->first;
    size_t idx_j = key_idx_[key_j];
    if (idx_j > idx_i) {
      sum += it->second * (idx_j > idx_l ? recover(key_l, key_j).transpose() : recover(key_j, key_l));
    }
  }
  return sum;
}

Matrix FastMarginals::recover(const Key &key_i, const Key &key_l) {
  KeyPair key_pair = std::make_pair(key_i, key_l);
  auto entry_iter = cov_cache_.entries.find(key_pair);
  if (entry_iter == cov_cache_.entries.end()) {
    Matrix res;
    if (key_i == key_l)
      res = getKeyDiag(key_l) * (getKeyDiag(key_l).transpose() - sumJ(key_l, key_l));
    else
      res = -getKeyDiag(key_i) * sumJ(key_l, key_i);

    cov_cache_.entries.insert(std::make_pair(key_pair, res));
    return res;
  } else {
    return entry_iter->second;
  }
}

}  // namespace gtsam
