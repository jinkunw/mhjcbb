#pragma once
#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <Eigen/Cholesky>

#include <priority_queue>
#include <stack>

namespace gtsam {
namespace da {

using symbol_shorthand::L;
using symbol_shorthand::X;

double chi2inv(double P, unsigned int dim);

struct Innovation {
  typedef std::shared_ptr<Innovation> shared_ptr;
  Key l;
  Vector error;
  Matrix Hx;
  Matrix Hl;
  Vector sigmas;
  double md;
};

template <typename POSE, typename POINT,
          typename BEARING = typename Bearing<POSE, POINT>::result_type,
          typename RANGE = typename Range<POSE, POINT>::result_type>
class JCBB {
  typedef BearingRange<POSE, POINT> BearingRange;
  typedef BearingRangeFactor<POSE, POINT> BearingRangeFactor;

 public:
  JCBB(const NonlinearFactorGraph &graph, const Values &values, double prob)
      : values_(values), prob_(prob) {
    marginals_ = Marginals(graph, values_);
    for (Key key : values_.keys())
      if (symbolChr(key) == 'l') keys_.push_back(key);

    for (int x = 0;; ++x) {
      if (!values_.exists(X(x))) {
        assert(x > 0);
        x0_ = X(x - 1);
        pose0_ = values_.at<POSE>(x0_);
        break;
      }
    }
  }

  void add(BEARING measuredBearing, RANGE measuredRange,
           const SharedNoiseModel &model) {
    innovations_.push_back({});
    for (Key l : keys_) {
      POINT point = values_.at<POINT>(l);
      BearingRangeFactor factor(x0_, l, measuredBearing, measuredRange, model);

      Innovation::shared_ptr inn(new Innovation);
      inn->l = l;
      inn->error = factor.evaluateError(pose0_, point, inn->Hx, inn->Hl);
      inn->sigmas = model->sigmas();
      inn->md = model->distance(inn->error);

      if (jc_(inn)) {
        innovations_.back().push_back(inn);
      }
    }
  }

  KeyVector match() {
    KeyVector keys;
    keys.push_back(x0_);
    for (const std::vector<Innovation::shared_ptr> &obs_inn : innovations_) {
      for (const Innovation::shared_ptr &inn : obs_inn)
        if (std::find(keys.begin(), keys.end(), inn->l) == keys.end())
          keys.push_back(inn->l);
    }
    joint_marginals_ = marginals_.jointMarginalCovariance(keys);

    for (FastVector<Innovation::shared_ptr> &linn : innovations_)
      std::sort(linn.begin(), linn.end(),
                [](Innovation::shared_ptr lhs, Innovation::shared_ptr rhs) {
                  return lhs->md < rhs->md;
                });

    jcbb({});

    KeyVector matched_keys;
    int new_l = 0;
    for (Innovation::shared_ptr &inn : best_hypothesis_)
      matched_keys.push_back(inn ? inn->l : keys_.size() + new_l++);
    return matched_keys;
  }

 private:
  void jcbb(const FastVector<Innovation::shared_ptr> &hypothesis) {
    int k = hypothesis.size();
    int h = pairings(hypothesis);
    if (k == innovations_.size()) {
      if (best_hypothesis_.empty() || h > pairings(best_hypothesis_))
        best_hypothesis_ = hypothesis;
      return;
    }

    FastSet<Key> existing;
    for (const Innovation::shared_ptr &inn : hypothesis)
      if (inn) existing.insert(inn->l);

    for (Innovation::shared_ptr &inn : innovations_[k]) {
      if (existing.find(inn->l) != existing.end()) continue;

      FastSet<Key> remaining;
      for (int j = k + 1; j < innovations_.size(); ++j) {
        for (Innovation::shared_ptr &future_inn : innovations_[j]) {
          if (future_inn->l != inn->l &&
              existing.find(future_inn->l) == existing.end())
            remaining.insert(future_inn->l);
        }
      }
      int max_remaining =
          std::min(remaining.size(), innovations_.size() - k - 1);
      if (h + 1 + max_remaining <= pairings(best_hypothesis_)) continue;

      FastVector<Innovation::shared_ptr> extended = hypothesis;
      extended.push_back(inn);
      if (jc(extended)) jcbb(extended);
    }
    FastSet<Key> remaining;
    for (int j = k + 1; j < innovations_.size(); ++j) {
      for (Innovation::shared_ptr &future_inn : innovations_[j]) {
        if (existing.find(future_inn->l) == existing.end())
          remaining.insert(future_inn->l);
      }
    }
    int max_remaining = std::min(remaining.size(), innovations_.size() - k - 1);
    if (best_hypothesis_.empty() ||
        h + max_remaining > pairings(best_hypothesis_)) {
      FastVector<Innovation::shared_ptr> extended = hypothesis;
      extended.push_back(nullptr);
      jcbb(extended);
    }
  }

  int pairings(const FastVector<Innovation::shared_ptr> &hypothesis) const {
    return std::count_if(
        hypothesis.cbegin(), hypothesis.cend(),
        [](const Innovation::shared_ptr &inn) { return inn != nullptr; });
  }

  bool jc_(const Innovation::shared_ptr &inn) const {
    Matrix S = Matrix::Zero(5, 5);
    S.block<3, 3>(0, 0) = marginals_.marginalCovariance(x0_);
    S.block<2, 2>(3, 3) = marginals_.marginalCovariance(inn->l);

    Matrix H(2, 5);
    H.block<2, 3>(0, 0) = inn->Hx;
    H.block<2, 2>(0, 3) = inn->Hl;

    Matrix R = inn->sigmas.asDiagonal();
    Vector e = inn->error;

    Matrix C = H * S * H.transpose() + R * R;
    double chi2 = e.transpose() * C.llt().solve(e);
    return chi2 < chi2inv(prob_, 2);
  }

  bool jc(const FastVector<Innovation::shared_ptr> &hypothesis) const {
    if (hypothesis.size() <= 1) return true;

    int XD = POSE::dimension, LD = POINT::dimension,
        FD = BearingRange::dimension;
    int N = XD, M = 0;
    KeyVector keys;
    keys.push_back(x0_);
    for (const Innovation::shared_ptr &inn : hypothesis) {
      if (!inn) continue;
      keys.push_back(inn->l);
      N += LD;
      M += FD;
    }

    Matrix S(N, N);
    for (int i = 0, p = 0; i < keys.size(); ++i) {
      Key ki = keys[i];
      int pi = values_.at(ki).dim();
      for (int j = i, q = p; j < keys.size(); ++j) {
        Key kj = keys[j];
        int qj = values_.at(kj).dim();
        S.block(p, q, pi, qj) = joint_marginals_.at(ki, kj);
        q += qj;
      }
      p += pi;
    }
    S.triangularView<Eigen::Lower>() = S.transpose();

    Matrix H = Matrix::Zero(M, N);
    Matrix R = Matrix::Zero(M, M);
    Vector e(M);
    for (int i = 0, j = 0; i < hypothesis.size(); ++i) {
      if (!hypothesis[i]) continue;
      H.block(j * FD, 0, FD, XD) = hypothesis[i]->Hx;
      H.block(j * FD, XD + j * LD, FD, LD) = hypothesis[i]->Hl;
      R.block(j * FD, j * FD, FD, FD) = hypothesis[i]->sigmas.asDiagonal();
      e.segment(j * FD, FD) = hypothesis[i]->error;
      j += 1;
    }

    Matrix C = H * S * H.transpose() + R * R;
    double chi2 = e.transpose() * C.llt().solve(e);
    return chi2 < chi2inv(prob_, M);
  }

 private:
  double prob_;

  FastVector<FastVector<Innovation::shared_ptr>> innovations_;
  FastVector<Innovation::shared_ptr> best_hypothesis_;

  KeyVector keys_;
  Marginals marginals_;
  JointMarginal joint_marginals_;
  Values values_;
  Key x0_;
  POSE pose0_;
};

template <typename POSE, typename POINT,
          typename BEARING = typename Bearing<POSE, POINT>::result_type,
          typename RANGE = typename Range<POSE, POINT>::result_type>
class MHJCBB {
  typedef BearingRange<POSE, POINT> BearingRange;
  typedef BearingRangeFactor<POSE, POINT> BearingRangeFactor;

  struct MatchInfo {
    int track;
    std::vector<Innovation> hypothesis;
    int num_pairings;
    double md;
    POSE pose;
    Matrix covariance;
  };

  struct MatchInfoCmp {
    bool operator()(const MatchInfo &a, const MatchInfo &b) const {
      return (a.num_pairings > b.num_pairings) || ((a.num_pairings == b.num_pairings) && (a.md < b.md));
    }
  };

  struct TrackInfo {
    FastVector<FastVector<Innovation::shared_ptr>> innovations;
    KeyVector keys;
    Marginals marginals;
    JointMarginal joint_marginals;
    Values values;
    Key x0;
    POSE pose0;

    std::stack<MatchInfo> stack;
  };

 public:
  MHJCBB(int max_tracks, double prob) : max_tracks_(max_tracks), prob_(prob) {}

  void initialize(const NonlinearFactorGraph &graph, const Values &values) {
    tracks_.push_back(TrackInfo());
    TrackInfo &track = tracks_.back();
    track.values = values;
    track.marginals = Marginals(graph, values);
    for (Key key : values.keys())
      if (symbolChr(key) == 'l') track.keys.push_back(key);

    for (int x = 0;; ++x) {
      if (!values.exists(X(x))) {
        assert(x > 0);
        track.x0 = X(x - 1);
        track.pose0 = values.at<POSE>(track.x0);
        break;
      }
    }
    track.stack.push(MatchInfo());
  }

  void add(BEARING measuredBearing, RANGE measuredRange,
           const SharedNoiseModel &model) {
    for (TrackInfo &track : tracks_) {
      track.innovations.push_back({});
      for (Key l : track.keys) {
        POINT point = track.values.template at<POINT>(l);
        BearingRangeFactor factor(track.x0, l, measuredBearing, measuredRange, model);

        Innovation::shared_ptr inn(new Innovation);
        inn->l = l;
        inn->error = factor.evaluateError(track.pose0, point, inn->Hx, inn->Hl);
        inn->sigmas = model->sigmas();
        inn->md = model->distance(inn->error);

        if (jc_(track, inn)) {
          track.innovations.back().push_back(inn);
        }
      }
    }
  }

  KeyVector match() {
    for (TrackInfo &track : tracks_) {
      KeyVector keys;
      keys.push_back(track.x0);
      for (const std::vector<Innovation::shared_ptr> &obs_inn : innovations_) {
        for (const Innovation::shared_ptr &inn : obs_inn)
          if (std::find(keys.begin(), keys.end(), inn->l) == keys.end())
            keys.push_back(inn->l);
      }
      track.joint_marginals = track.marginals.jointMarginalCovariance(keys);

      for (FastVector<Innovation::shared_ptr> &linn : track.innovations)
        std::sort(linn.begin(), linn.end(),
                  [](Innovation::shared_ptr lhs, Innovation::shared_ptr rhs) {
                    return lhs->md < rhs->md;
                  });
    }

    mhjcbb({});

    KeyVector matched_keys;
    int new_l = 0;
    for (Innovation::shared_ptr &inn : best_hypothesis_)
      matched_keys.push_back(inn ? inn->l : keys_.size() + new_l++);
    return matched_keys;
  }

 private:
  void mhjcbb() {
    int tracks_done = 0;
    while (tracks_done < max_tracks) {
      for (TrackInfo &ti : tracks_) {
        tracks_done += jcbb(ti);
      }
    }
    prune();
  }

  bool jcbb(TrackInfo &ti) {
    while (!ti.stack.empty()) {
      MatchInfo mi = ti.stack.top();
      ti.stack.pop();

      int k = mi.hypothesis.size();
      int h = pairings(ti.hypothesis);
      if (k == ti.innovations.size()) {
        if (jc(ti, mi) && screen(mi)) {
          if (best_hypotheses_.size() == max_tracks_)
            best_hypotheses_.pop();
          best_hypotheses_.push(mi);
          return false;
        }
      } else {
        FastSet<Key> existing;
        for (const Innovation::shared_ptr &inn : mi.hypothesis)
          if (inn) existing.insert(inn->l);

        for (Innovation::shared_ptr &inn : ti.innovations[k]) {
          if (existing.find(inn->l) != existing.end()) continue;

          FastSet<Key> remaining;
          for (int j = k + 1; j < ti.innovations.size(); ++j) {
            for (Innovation::shared_ptr &future_inn : ti.innovations[j]) {
              if (future_inn->l != inn->l &&
                  existing.find(future_inn->l) == existing.end())
                remaining.insert(future_inn->l);
            }
          }
          int max_remaining =
              std::min(remaining.size(), ti.innovations.size() - k - 1);
          int future_pairings = h + 1 + max_remaining;
          if (!best_hypotheses_.empty()) {
            int min_pairings = best_hypotheses_.top().num_pairings;
            if (future_pairings < min_pairings)
              continue;
            if (best_hypotheses_.size() == ti.innovations.size() &&
                future_pairings == min_pairings && mi.md > best_hypotheses.top().md)
              continue;
          }

          MatchInfo extended = mi;
          extended.hypothesis.push_back(inn);
          extended.num_pairings += 1;
          if (jc(ti, mi))
            ti.stack.push(extended);
        }
        FastSet<Key> remaining;
        for (int j = k + 1; j < ti.innovations.size(); ++j) {
          for (Innovation::shared_ptr &future_inn : ti.innovations[j]) {
            if (existing.find(future_inn->l) == existing.end())
              remaining.insert(future_inn->l);
          }
        }
        int max_remaining = std::min(remaining.size(), ti.innovations.size() - k - 1);
        if (best_hypothesis_.empty() ||
            h + max_remaining >= best_hypotheses_.top().num_pairings) {
          MatchInfo extended = mi;
          extended.hypothesis.push_back(nullptr);
          ti.stack.push(extended);
        }
      }
    }
    return true;
  }

  bool screen(const MatchInfo &mi) {
  }

  int pairings(const FastVector<Innovation::shared_ptr> &hypothesis) const {
    return std::count_if(
        hypothesis.cbegin(), hypothesis.cend(),
        [](const Innovation::shared_ptr &inn) { return inn != nullptr; });
  }

  bool jc_(const TrackInfo &ti, const Innovation::shared_ptr &inn) const {
    Matrix S = Matrix::Zero(5, 5);
    S.block<3, 3>(0, 0) = ti.marginals.marginalCovariance(x0_);
    S.block<2, 2>(3, 3) = ti.marginals.marginalCovariance(inn->l);

    Matrix H(2, 5);
    H.block<2, 3>(0, 0) = inn->Hx;
    H.block<2, 2>(0, 3) = inn->Hl;

    Matrix R = inn->sigmas.asDiagonal();
    Vector e = inn->error;

    Matrix C = H * S * H.transpose() + R * R;
    double chi2 = e.transpose() * C.llt().solve(e);
    return chi2 < chi2inv(prob_, 2);
  }

  bool jc(const TrackInfo &ti, MatchInfo &mi) const {
    if (hypothesis.size() <= 1) return true;

    int XD = POSE::dimension, LD = POINT::dimension,
        FD = BearingRange::dimension;
    int N = XD, M = 0;
    KeyVector keys;
    keys.push_back(ti.x0);
    for (const Innovation::shared_ptr &inn : mi.hypothesis) {
      if (!inn) continue;
      keys.push_back(inn->l);
      N += LD;
      M += FD;
    }

    Matrix S(N, N);
    for (int i = 0, p = 0; i < keys.size(); ++i) {
      Key ki = keys[i];
      int pi = ti.values.at(ki).dim();
      for (int j = i, q = p; j < keys.size(); ++j) {
        Key kj = keys[j];
        int qj = ti.values.at(kj).dim();
        S.block(p, q, pi, qj) = ti.joint_marginals.at(ki, kj);
        q += qj;
      }
      p += pi;
    }
    S.triangularView<Eigen::Lower>() = S.transpose();

    Matrix H = Matrix::Zero(M, N);
    Matrix R = Matrix::Zero(M, M);
    Vector e(M);
    for (int i = 0, j = 0; i < mi.hypothesis.size(); ++i) {
      if (!mi.hypothesis[i]) continue;
      H.block(j * FD, 0, FD, XD) = mi.hypothesis[i]->Hx;
      H.block(j * FD, XD + j * LD, FD, LD) = mi.hypothesis[i]->Hl;
      R.block(j * FD, j * FD, FD, FD) = mi.hypothesis[i]->sigmas.asDiagonal();
      e.segment(j * FD, FD) = mi.hypothesis[i]->error;
      j += 1;
    }

    Matrix C = H * S * H.transpose() + R * R;
    double chi2 = e.transpose() * C.llt().solve(e);
    return chi2 < chi2inv(prob_, M);
  }

 private:
  int max_tracks_;
  double prob_;

  FastVector<TrackInfo> tracks_;
  std::priority_queue<MatchInfo, std::vector<MatchInfo>, MatchInfoCmd> best_hypotheses_;
};

double normalCDF(double u) {
  static const double a[5] = {1.161110663653770e-002, 3.951404679838207e-001,
                              2.846603853776254e+001, 1.887426188426510e+002,
                              3.209377589138469e+003};
  static const double b[5] = {1.767766952966369e-001, 8.344316438579620e+000,
                              1.725514762600375e+002, 1.813893686502485e+003,
                              8.044716608901563e+003};
  static const double c[9] = {
      2.15311535474403846e-8, 5.64188496988670089e-1, 8.88314979438837594e00,
      6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02,
      1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725E03};
  static const double d[9] = {
      1.00000000000000000e00, 1.57449261107098347e01, 1.17693950891312499e02,
      5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03,
      4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};
  static const double p[6] = {1.63153871373020978e-2, 3.05326634961232344e-1,
                              3.60344899949804439e-1, 1.25781726111229246e-1,
                              1.60837851487422766e-2, 6.58749161529837803e-4};
  static const double q[6] = {1.00000000000000000e00, 2.56852019228982242e00,
                              1.87295284992346047e00, 5.27905102951428412e-1,
                              6.05183413124413191e-2, 2.33520497626869185e-3};
  double y, z;

  y = fabs(u);
  // clang-format off
  if (y <= 0.46875 * 1.4142135623730950488016887242097) {
    /* evaluate erf() for |u| <= sqrt(2)*0.46875 */
    z = y * y;
    y = u * ((((a[0] * z + a[1]) * z + a[2]) * z + a[3]) * z + a[4]) / ((((b[0] * z + b[1]) * z + b[2]) * z + b[3]) * z + b[4]);
    return 0.5 + y;
  }

  z = exp(-y * y / 2) / 2;
  if (y <= 4.0) {
    /* evaluate erfc() for sqrt(2)*0.46875 <= |u| <= sqrt(2)*4.0 */
    y = y / 1.4142135623730950488016887242097;
    y = ((((((((c[0] * y + c[1]) * y + c[2]) * y + c[3]) * y + c[4]) * y + c[5]) * y + c[6]) * y + c[7]) * y + c[8]) / ((((((((d[0] * y + d[1]) * y + d[2]) * y + d[3]) * y + d[4]) * y + d[5]) * y + d[6]) * y + d[7]) * y + d[8]);
    y = z * y;
  } else {
    /* evaluate erfc() for |u| > sqrt(2)*4.0 */
    z = z * 1.4142135623730950488016887242097 / y;
    y = 2 / (y * y);
    y = y * (((((p[0] * y + p[1]) * y + p[2]) * y + p[3]) * y + p[4]) * y + p[5]) / (((((q[0] * y + q[1]) * y + q[2]) * y + q[3]) * y + q[4]) * y + q[5]);
    y = z * (0.564189583547756286948 - y);
  }
  return (u < 0.0 ? y : 1 - y);
}

double normalQuantile(double p) {
  double q, t, u;

  static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
  static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                              -2.400758277161838e+00, -2.549732539343734e+00,
                              4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                              2.445134137142996e+00, 3.754408661907416e+00};

  q = std::min(p, 1 - p);

  if (q > 0.02425) {
    /* Rational approximation for central region. */
    u = q - 0.5;
    t = u * u;
    u = u * (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1);
  } else {
    /* Rational approximation for tail region. */
    t = sqrt(-2 * log(q));
    u = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1);
  }

  /* The relative error of the approximation has absolute value less
  than 1.15e-9.  One iteration of Halley's rational method (third
  order) gives full machine precision... */
  t = normalCDF(u) - q;                                      /* error */
  t = t * 2.506628274631000502415765284811 * exp(u * u / 2); /* f(u)/df(u) */
  u = u - t / (1 + u * t / 2); /* Halley's method */

  return (p > 0.5 ? -u : u);
}

double chi2inv(double P, unsigned int dim) {
  if (P == 0)
    return 0;
  else
    return dim * pow(1.0 - 2.0 / (9 * dim) + sqrt(2.0 / (9 * dim)) * normalQuantile(P), 3);
}
}  // namespace da
}  // namespace gtsam