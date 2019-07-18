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
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/array_utils.hpp"
#include "Utils/kronecker_product.hpp"
#include "Utils/next_variation.hpp"
#include "abstract_operator.hpp"
// clang-format on

namespace netket {

// struct DeOp {
//   // Adjusts the occ and unocc vectors during loops to account for a
//   // destruction operator being applied.
//
//   int q_;
//   std::vector<int> &occ;
//   std::vector<int> &unocc;
//
//   // Modifies occ and unocc
//   DeOp(int q, std::vector<int> &occ, std::vector<int> &unocc)
//       : q_(q), occ(occ), unocc(unocc) {
//     // Remove s from occ
//     int q_ind;
//     for (int i = 0; i < occ.size(); i++) {
//       if (occ[i] == q_) {
//         q_ind = i;
//         break;
//       }
//     }
//     occ.erase(occ.begin() + q_ind);
//
//     // Add s to unocc
//     unocc.push_back(q_);
//   }
//
//   ~DeOp() {
//     // Add s to occ
//     occ.push_back(q_);
//
//     // Remove s from unocc
//     // unocc.pop_back();
//     int q_ind;
//     for (int i = 0; i < unocc.size(); i++) {
//       if (unocc[i] == q_) {
//         q_ind = i;
//         break;
//       }
//     }
//     unocc.erase(unocc.begin() + q_ind);
//   }
// };
//
// struct CrOp {
//   // Adjusts the occ and unocc vectors during loops to account for a
//   // creation operator being applied.
//   int p_;
//   std::vector<int> &occ;
//   std::vector<int> &unocc;
//
//   CrOp(int p, std::vector<int> &occ, std::vector<int> &unocc)
//       : p_(p), occ(occ), unocc(unocc) {
//     // Remove p from unocc
//     int p_ind;
//     for (int i = 0; i < unocc.size(); i++) {
//       if (unocc[i] == p_) {
//         p_ind = i;
//         break;
//       }
//     }
//     unocc.erase(unocc.begin() + p_ind);
//
//     // Add p to occ
//     occ.push_back(p_);
//   }
//   ~CrOp() {
//     // Add p back to unocc
//     unocc.push_back(p_);
//     // Remove p from occ
//     // occ.pop_back();
//     int p_ind;
//     for (int i = 0; i < occ.size(); i++) {
//       if (occ[i] == p_) {
//         p_ind = i;
//         break;
//       }
//     }
//     occ.erase(occ.begin() + p_ind);
//   }
// };

struct OneBodyIntegrals {
  const Eigen::MatrixXd integrals;
  const int N;

  OneBodyIntegrals(const Eigen::MatrixXd h) : integrals(h), N(h.cols()) {
    std::cout << "Initializing OneBodyIntegrals\n";
  }

  double operator()(int i, int j) const {
    // Indices given in spin orbital basis and the integrals are referenced in
    // spatial orbital basis
    return integrals(i / 2, j / 2);
  }
};

struct TwoBodyIntegrals {
  const Eigen::MatrixXd integrals;
  const int N;

  TwoBodyIntegrals(const Eigen::MatrixXd g) : integrals(g), N(sqrt(g.cols())) {
    std::cout << "Initializing TwoBodyIntegrals\n";
  }

  double operator()(int i, int j, int k, int l) const {
    // Indices given in spin orbital basis and the integrals are referenced in
    // spatial orbital basis
    return integrals(N * i / 2 + j / 2, N * k / 2 + l / 2);
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
  const double constant_;

  // static constexpr double mel_cutoff_ = 1.0e-6;

public:
  explicit QCHamiltonian(std::shared_ptr<const AbstractHilbert> hilbert,
                         const Eigen::MatrixXd &h, const Eigen::MatrixXd &g,
                         double e0)
      : AbstractOperator(hilbert), h_(h), g_(g), norb_(hilbert->Size()),
        nmo_(hilbert->Size()), constant_(e0) {
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
    std::vector<double> conf(v.data(), v.data() + v.size());

    for (int vi = 0; vi < v.size(); vi++) {
      std::cout << v(vi) << " ";
    }
    std::cout << "\n";
    double e_HF = Calculate_Hij(occ);
    std::cout << "HF energy " << e_HF << "\n";
    exit(0);

    // Generate Configurations
    for (auto p : unocc) {
      for (auto q : occ) {
        // Hij_1Excite
        continue;
      }
    }

    // One-body excitations
    // for (auto q : occ) {
    //   // DeOp dq = DeOp(q, occ, unocc);
    //   mel[0] +=
    //       h_(q / 2, q / 2); // + g_(q / 2 * nmo_ + q / 2, q / 2 * nmo_ + q /
    //       2);
    //
    //       for (auto p : unocc) {
    //         // Skip if i and j don't both act on up (or down) spin
    //         if (p % 2 != q % 2) {
    //           continue;
    //         }
    //         if (p == q) { // Diagonal has already been handled
    //           continue;
    //         }
    //         // CrOp cp = CrOp(p, occ, unocc);
    //         // TODO use occ, unocc for parity (maybe not because the
    //         occupation
    //         // for
    //         // the right most operator would be messed up) but think about it
    //         // Order of the connectors and new values should match the order
    //         in
    //         // which the operators are applied
    //         connectors.push_back({q, p});
    //         newconfs.push_back({0., 1.});
    //         mel.push_back(Parity(v, p, q) * h_(p / 2, q / 2));
    //       }
    // }

    // Two-body excitations
    // two body integrals from PySCF (i.e. eris) are stored as (pq|rs)
    // The two body part of the hamiltonian is
    // (pq|rs) p^\dagger q r^\dagger s

    // for (auto s : occ) { // Annihilation
    //   mel[0] += g_(s / 2 * nmo_ + s / 2, s / 2 * nmo_ + s / 2);
    //
    //   DeOp ds = DeOp(s, occ, unocc);
    //   for (auto r : unocc) { // Creation
    //     if (s % 2 != r % 2) {
    //       continue;
    //     }
    //     CrOp cr = CrOp(r, occ, unocc);
    //     for (auto q : occ) { // Creation
    //       DeOp dq = DeOp(q, occ, unocc);
    //       for (auto p : unocc) { // Annihilation
    //         // Diagonal taken care of in one-body double for-loop
    //         if (p == q && q == r && r == s) {
    //           continue;
    //         }
    //         if (p % 2 != q % 2) {
    //           continue;
    //         }
    //         CrOp cp = CrOp(p, occ, unocc);
    //
    //         // Order of the connectors and new values should match the order
    //         // in which the operators are applied
    //         connectors.push_back({s, r, q, p});
    //         newconfs.push_back({0., 1., 0., 1.});
    //         mel.push_back(Parity(v, p, q, r, s) *
    //                       g_(p / 2 * nmo_ + q / 2, r / 2 * nmo_ + s / 2));
    //       }
    //     }
    //   }
    // } // End two body

    if (occ.size() != 12) { // TODO for debugging only
      throw "WE LOST SOME ELECTRONS";
    }
  } // End FindConn

  // Excitation = 0 i.e. same configurations
  // Returns the energy of the configuration
  double Calculate_Hij(std::vector<int> &occ) const {
    double Hij = 0.0;

    double onebody = 0.0;
    double twobody = 0.0;

    for (auto i : occ) {
      Hij += h_(i, i);
      onebody += h_(i, i);

      for (int ji = i + 1; ji < occ.size(); ji++) {
        int j = occ[ji];
        // Direct
        Hij += g_(i, j, i, j);
        twobody += g_(i, j, i, j);
        if (i % 2 == j % 2) {
          // Exchange
          Hij -= g_(i, j, j, i);
          twobody -= g_(i, j, j, i);
        }
      }
    }

    std::cout << "OneBody " << onebody << "\n";
    std::cout << "TwoBody " << twobody << "\n";
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
