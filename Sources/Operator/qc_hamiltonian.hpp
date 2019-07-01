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
#include "Hilbert/abstract_hilbert.hpp"
#include "Utils/array_utils.hpp"
#include "Utils/kronecker_product.hpp"
#include "Utils/next_variation.hpp"
#include "abstract_operator.hpp"
// clang-format on

namespace netket {

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
  const Eigen::MatrixXd h_;
  const Eigen::MatrixXd g_;
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
    if (norb_ != h_.rows() * 2 || norb_ * norb_ != g_.rows() * 4) {
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
    std::vector<double> current_conf(v.data(), v.data() + v.size());

    // One-body excitations
    for (auto q : unocc) {
      mel[0] += h_(q / 2, q / 2);

      // Off diagonal excitations
      for (auto p : occ) {
        if (h_(p / 2, q / 2) > 0) {
          // Skip if i and j don't both act on up (or down) spin
          if (p % 2 != q % 2) {
            continue;
          }
          connectors.push_back({p, q});
          newconfs.push_back({1., 0.});
          mel.push_back(Parity(v, p, q) * h_(p / 2, q / 2));
        }
      }
    }

    // Two-body excitations
    // two body integrals from PySCF (i.e. eris) are stored as (pq|rs)
    // The two body part of the hamiltonian is
    // (pq|rs) p^\dagger q r^\dagger s
    for (auto q : unocc) {
      mel[0] += g_(q / 2 * nmo_ + q, q / 2 * nmo_ + q); // All diag

      for (auto p : occ) {
        if (p % 2 != q % 2) {
          continue;
        }

        // Partial diagonal
        connectors.push_back({p, q});
        newconfs.push_back({1., 0.});
        mel.push_back(0.);
        Complex &partial_diag_mel = mel.back();

        for (auto r : occ) {
          if (r == p) {
            continue;
          }
          // Partial diagonal
          partial_diag_mel +=
              Parity(v, p, q) * g_(p / 2 * nmo_ + q, r / 2 * nmo_ + r);
          partial_diag_mel +=
              Parity(v, p, q) * g_(r / 2 * nmo_ + r, p / 2 * nmo_ + q);

          // All indices are different
          // p != q != r != s
          for (auto s : unocc) {
            if (s == q) {
              continue;
            }
            if (r % 2 != s % 2) {
              continue;
            }
            if (g_(p / 2 * nmo_ + q, r / 2 * nmo_ + s) > 0) {
              connectors.push_back({p, q, r, s});
              newconfs.push_back({1., 0., 1., 0.});
              mel.push_back(Parity(v, p, q, r, s) *
                            g_(p / 2 * nmo_ + q, r / 2 * nmo_ + s));
            }
          }
        }
      }
    } // End two body
  }   // End FindConn

  void GetOpenClosed(VectorConstRefType v, std::vector<int> &occ,
                     std::vector<int> &unocc) const {
    // Find occ and unocc spin orbitals
    for (int i = 0; i < norb_; i++) {
      if (v(i) == 0) { // unoccupied
        occ.push_back(i);
      } else { // occupied
        unocc.push_back(i);
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
