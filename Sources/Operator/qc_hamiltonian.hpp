// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_QC_HAMILTONIAN_HPP
#define NETKET_QC_HAMILTONIAN_HPP

// clang-format off
#include <mpi.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <vector>
#include <math.h>
#include <stdio.h>
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/array_utils.hpp"
#include "Utils/kronecker_product.hpp"
#include "Utils/next_variation.hpp"
#include "abstract_operator.hpp"
// clang-format on

namespace netket {
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct OneBodyIntegrals {
  const RowMatrixXd integrals;
  const int N;

  OneBodyIntegrals(const RowMatrixXd h) : integrals(h), N(h.cols()) {
    std::cout << "Initializing OneBodyIntegrals\n";
  }

  double operator()(int i, int j) const {
    // Indices given in spin orbital basis and the integrals are referenced in
    // spatial orbital basis
    return integrals(i / 2, j / 2);
  }
};

struct TwoBodyIntegrals {
  const RowMatrixXd integrals;
  const int N;

  TwoBodyIntegrals(const RowMatrixXd g) : integrals(g), N(sqrt(g.cols())) {
    std::cout << "Initializing TwoBodyIntegrals\n";
  }

  double operator()(int i, int j, int k, int l) const {
    // Indices given in spin orbital basis and the integrals are referenced in
    // spatial orbital basis
    return integrals(N * (i / 2) + (j / 2), N * (k / 2) + (l / 2));
  }
};

/**
    Class for local operators acting on a list of sites and for generic local
    Hilbert spaces.
*/

class QCHamiltonian : public AbstractOperator {
public:
  using MelType = Complex;
  using MatType = std::vector<std::vector<MelType>>;
  using SiteType = std::vector<int>;
  using MapType = std::map<std::vector<double>, int>;
  using StateType = std::vector<std::vector<double>>;
  using ConnType = std::vector<std::vector<int>>;
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

private:
  // const AbstractHilbert &hilbert_;
  std::vector<MatType> mat_;
  std::vector<SiteType> sites_;

  std::vector<MapType> invstate_;
  std::vector<StateType> states_;
  std::vector<ConnType> connected_;

  std::size_t nops_;

  // QC Stuff
  const OneBodyIntegrals h_;
  const TwoBodyIntegrals g_;
  const int norb_;
  const int nmo_;
  const int nelec_;
  const double constant_;

  static constexpr double mel_cutoff_ = 1.0e-8;

public:
  explicit QCHamiltonian(std::shared_ptr<const AbstractHilbert> hilbert,
                         const RowMatrixXd &h, const RowMatrixXd &g, double e0,
                         int nelec)
      : AbstractOperator(hilbert), h_(h), g_(g), norb_(hilbert->Size()),
        nmo_(hilbert->Size()), nelec_(nelec), constant_(e0) {
    // Check that dimension of one and two body integrals match dimension of the
    // hilbert space
    if (norb_ != h_.N * 2 || norb_ != g_.N * 2) {
      throw InvalidInputError(
          "Matrix size in operator is inconsistent with Hilbert space");
    }

    InfoMessage() << "Quantum chemistry Hamiltonian created" << std::endl;
  }

  void FindConn(VectorConstRefType v, std::vector<Complex> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override {
    // He we (NetKet) have chosen to store the information that requires no
    // connection information, i.e. the diagonal at the first index (mel[0]).
    // We also choose to work with the convention that spins are stored
    // \alpha \beta \alpha ... i.e. alphas are even spin orbitals and betas
    assert(v.size() == hilbert_.Size());

    connectors.clear();
    newconfs.clear();
    mel.clear();

    connectors.resize(1);
    newconfs.resize(1);
    mel.resize(1);

    // mel[0] = constant_;
    mel[0] = 0;
    connectors[0].resize(0);
    newconfs[0].resize(0);

    std::vector<int> occ;   // Keep track of occ spin orbitals
    std::vector<int> unocc; // Keep trach of unocc spin orbitals
    GetOpenClosed(v, occ, unocc);

    // Excitation = 0 (no new configurations generated)
    mel[0] += Calculate_Hij(occ);

    // Excitation = 1
    for (auto p : unocc) {
      for (auto q : occ) {
        if (p % 2 != q % 2) {
          continue;
        }
        double Hij = Calculate_Hij(occ, p, q);
        if (std::abs(Hij) > mel_cutoff_) {
          continue;
        }
        connectors.push_back({q, p});
        newconfs.push_back({0., 1.});
        mel.push_back(Hij);
      }
    }

    std::vector<std::vector<int>> dets;
    GetCIDDets(occ, unocc, dets);
    for (auto det : dets) {
      int p = det.at(3);
      int q = det.at(2);
      int r = det.at(1);
      int s = det.at(0);

      double Hij = Calculate_Hij(occ, v, p, q, r, s);
      if (std::abs(Hij) > mel_cutoff_) {
        continue;
      }
      connectors.push_back({s, r, q, p});
      newconfs.push_back({0., 1., 0., 1.});
      mel.push_back(Calculate_Hij(occ, v, p, q, r, s));
    }

    if (occ.size() != nelec_) { // TODO for debugging only
      throw "WE LOST SOME ELECTRONS";
    }
  } // End FindConn

  void GetCIDDets(std::vector<int> &occ, std::vector<int> &unocc,
                  std::vector<std::vector<int>> &dets) const {
    std::vector<int> occAlpha;
    std::vector<int> occBeta;
    std::vector<int> unoccAlpha;
    std::vector<int> unoccBeta;

    for (auto o : occ) {
      if (o % 2 == 0) {
        occAlpha.push_back(o);
      } else if (o % 2 == 1) {
        occBeta.push_back(o);
      }
    }

    for (auto u : unocc) {
      if (u % 2 == 0) {
        unoccAlpha.push_back(u);
      } else if (u % 2 == 1) {
        unoccBeta.push_back(u);
      }
    }

    // Three options here:
    // 1) One alpha excitation and one beta
    // 2) Two alpha excitations
    // 3) Two beta excitations

    // 1)
    for (auto oa : occAlpha) {
      for (auto ua : unoccAlpha) {
        for (auto ob : occBeta) {
          for (auto ub : unoccBeta) {
            dets.push_back({ob, ub, oa, ua});
          }
        }
      }
    }

    // 2)
    for (int i1 = 0; i1 < occAlpha.size(); i1++) {
      for (int i2 = i1 + 1; i2 < occAlpha.size(); i2++) {
        for (int j1 = 0; j1 < unoccAlpha.size(); j1++) {
          for (int j2 = j1 + 1; j2 < unoccAlpha.size(); j2++) {
            int p = occAlpha.at(i1);
            int q = occAlpha.at(i2);
            int r = unoccAlpha.at(j1);
            int s = unoccAlpha.at(j2);

            dets.push_back({s, r, q, p});
          }
        }
      }
    }

    // 3)
    for (int i1 = 0; i1 < occBeta.size(); i1++) {
      for (int i2 = i1 + 1; i2 < occBeta.size(); i2++) {
        for (int j1 = 0; j1 < unoccBeta.size(); j1++) {
          for (int j2 = j1 + 1; j2 < unoccBeta.size(); j2++) {
            int p = occBeta.at(i1);
            int q = occBeta.at(i2);
            int r = unoccBeta.at(j1);
            int s = unoccBeta.at(j2);

            dets.push_back({s, r, q, p});
          }
        }
      }
    }
  }

  /**
   * @brief Returns the energy of the configuration specified by occ. This is
   * Case 1 in Tables 2.3/2.4 in Szabo and Ostlund (1989).
   *
   * @param occ The vector of occupied spin orbitals.
   * @return double Energy of the configuration.
   */
  double Calculate_Hij(std::vector<int> &occ) const {
    double Hij = 0.0;

    // std::cout << "Occupied orbitals ";
    // for (auto o : occ) {
    //   std::cout << o << " ";
    // }
    // std::cout << "\n";

    for (int ii = 0; ii < occ.size(); ii++) {
      int i = occ.at(ii);
      Hij += h_(i, i);
      // printf("h_(%i %i) = %f\n", i, i, h_(i, i));

      for (int ji = ii + 1; ji < occ.size(); ji++) {
        int j = occ.at(ji);
        // Direct
        Hij += g_(i, i, j, j);
        // printf("g_(%i %i %i %i) = %f\n", i, i, j, j, g_(i, i, j, j));
        if (i % 2 == j % 2) {
          // Exchange
          Hij -= g_(i, j, j, i);
          // printf("g_(%i %i %i %i) = %f\n", i, j, j, i, g_(i, j, j, i));
        }
      }
    }

    // std::cout << Hij << "\n";
    // printf("Hij = %f\n", Hij);

    return Hij;
  }

  /**
   * @brief Returns the Hamiltonian matrix element between two configurations
   * that differ by a single excitation operator (one creation and one
   * excitation operator). This is Case 2 in Tables 2.3/2.4 in Szabo and Ostlund
   * (1989).
   *
   * @param occ The vector of occupied spin orbitals.
   * @param p The spin orbital index of the creation operator.
   * @param q The spin orbital index of the annihilation operator.
   * @return double \f$\langle I|\hat{H}| J\rangle\f$
   */
  double Calculate_Hij(std::vector<int> &occ, int p, int q) const {
    double Hij = h_(p, q);
    double par = 1.0;

    for (auto i : occ) {
      if (i > std::min(p, q) && i < std::max(p, q)) {
        par *= -1.;
      }
      Hij += g_(p, q, i, i);
      if (q % 2 == i % 2) {
        Hij -= g_(p, i, i, q);
      }
    }

    return Hij * par;
  }

  /**
   * @brief Returns the Hamiltonian matrix element between two configurations
   * that differ by a double excitation operator (one creation and one
   * excitation operator). This is Case 3 in Tables 2.3/2.4 in Szabo and Ostlund
   * (1989).
   *
   * @param occ The vector of occupied spin orbitals.
   * @param v State vector for the ket.
   * @param p Orbital index for creation operator
   * @param q Orbital index for destruction operator
   * @param r Orbital index for creation operator
   * @param s Orbital index for destruction operator
   * @return double \f$\langle I|\hat{H}| J\rangle\f$
   */
  double Calculate_Hij(std::vector<int> &occ, VectorConstRefType v, int p,
                       int q, int r, int s) const {
    double Hij = 0;
    double par = Parity(v, p, q, r, s);

    Hij += g_(p, q, r, s);

    if (q % 2 == r % 2) {
      Hij -= g_(p, s, r, q);
    }

    Hij *= par;
    return Hij;
  }

  void GetOpenClosed(VectorConstRefType v, std::vector<int> &occ,
                     std::vector<int> &unocc) const {
    // Find occ and unocc spin orbitals
    for (int i = 0; i < norb_; i++) {
      if (v(i) == 0) { // unoccupied
        unocc.push_back(i);
      } else if (v(i) == 1) { // occupied
        occ.push_back(i);
      } else {
        std::cout << "Occupation number not supported for QCHamiltonian\n";
        throw "ERROR";
      }
    }
  }

  double Parity(VectorConstRefType v, int p, int q) const {
    // p^+ q
    // q -> p
    double sign = 1.0;

    // Annihilation operator
    for (int b = 0; b < q; b++) {
      if (v(b) == 1.) {
        sign *= -1.0;
      }
    }

    // Creation operator
    for (int a = 0; a < p; a++) {
      if (v(a) == 1.) {
        sign *= -1.0;
      }
    }

    if (p >= q) {
      sign *= -1.;
    }
    return sign;
  }

  double Parity(VectorConstRefType v, int p, int q, int r, int s) const {
    // p^+ q r^+ s
    // q -> p
    // s -> r
    double sign = 1.0;

    // S
    for (int d = 0; d < s; d++) {
      if (v(d) == 1.) {
        sign *= -1.0;
      }
    }

    // R^+
    for (int c = 0; c < r; c++) {
      if (v(c) == 1.) {
        sign *= -1.0;
      }
    }

    if (r >= s) {
      sign *= -1.;
    }

    // Q
    for (int b = 0; b < q; b++) {
      if (v(b) == 1.) {
        sign *= -1.0;
      }
    }

    // For the other two cases the sign doesn't change
    if ((q >= s && q < r) || (q < s && q >= r)) {
      sign *= -1.;
    }

    // P^+
    for (int a = 0; a < p; a++) {
      if (v(a) == 1.) {
        sign *= -1.0;
      }
    }

    // For the other four cases the sign doesn't change
    if ((p >= q && p >= r && p >= s) || (p >= q && p < r && p < s) ||
        (p < q && p >= r && p < s) || (p < q && p < r && p >= s)) {
      sign *= -1.;
    }
    return sign;
  }

  inline int StateNumber(VectorConstRefType v, int opn) const {
    // TODO use a mask instead of copies
    std::vector<double> state(sites_[opn].size());
    for (std::size_t i = 0; i < sites_[opn].size(); i++) {
      state[i] = v(sites_[opn][i]);
    }
    return invstate_[opn].at(state);
  }

  const std::vector<MatType> &LocalMatrices() const { return mat_; }
  const std::vector<SiteType> &ActingOn() const { return sites_; }

  // const AbstractHilbert &GetHilbert() const noexcept override {
  //   return hilbert_;
  // }

  std::size_t Size() const { return mat_.size(); }
}; // namespace netket

} // namespace netket
#endif
