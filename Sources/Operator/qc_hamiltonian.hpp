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

  double constant_;
  std::size_t nops_;

  // QC Stuff
  const Eigen::MatrixXd &h_;
  const Eigen::MatrixXd &g_;
  const int norb_;

  // static constexpr double mel_cutoff_ = 1.0e-6;

public:
  explicit QCHamiltonian(std::shared_ptr<const AbstractHilbert> hilbert,
                         const Eigen::MatrixXd &h, const Eigen::MatrixXd &g)
      : AbstractOperator(hilbert), h_(h), g_(g), norb_(hilbert->Size()) {
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

    mel[0] = constant_;
    connectors[0].resize(0);
    newconfs[0].resize(0);

    std::vector<int> open;   // Keep track of open spin orbitals
    std::vector<int> closed; // Keep trach of closed spin orbitals
    GetOpenClosed(v, open, closed);
    std::vector<double> current_conf(v.data(), v.data() + v.size());

    // TODO good up till here

    // One-body excitations
    // Iterate  over closed orbitals to generate annihilation operators the
    // iterate over open orbitals to generate creation operators
    // for

    // One-body excitations
    for (auto j : closed) {
      mel[0] += h_(j / 2, j / 2);

      // Off diagonal excitations
      for (auto i : open) {
        if (h_(i / 2, j / 2) > 0) {
          // Skip if i and j don't both act on up (or down) spin
          if (i % 2 != j % 2) {
            continue;
          }
          connectors.push_back({i, j});
          // Create new configuration
          std::vector<double> new_state = current_conf;
          new_state.at(j) = 0.;
          new_state.at(i) = 1.;
          newconfs.push_back(new_state);
          // mel.push_back(Parity(current_conf, i, j) * h_(i / 2, j / 2));
        }
      }
    }

    // Two-body excitations
    // a^\dagger_i a^\dagger_j a_k a_l
    // k -> j
    // l -> i
    // for (auto i : open) {
    //   for (auto l : closed) {
    //     if (i % 2 != l % 2) {
    //       continue;
    //     }
    //     for (auto k : closed) {
    //       // Diagonal in k
    //       connectors.push_back({i, l});
    //       std::vector<double> diag_conf = current_conf;
    //       diag_conf.at(l) = 0.;
    //       diag_conf.at(i) = 1.;
    //       newconfs.push_back(diag_conf);
    //       mel.push_back(Parity(current_conf, i, l) *
    //                     g_(i / 2 * norb_ + k / 2, k / 2 * norb_ + l / 2));
    //
    //       for (auto j : open) {
    //         if (j % 2 != k % 2) {
    //           continue;
    //         }
    //         //
    //         connectors.push_back({i, j, k, l});
    //         std::vector<double> conf(v.data(), v.data() + v.size());
    //         conf.at(l) = 0.;
    //         conf.at(i) = 1.;
    //         double sign = Parity(current_conf, i, l);
    //
    //         sign *= Parity(conf, j, k);
    //         newconfs.push_back(conf);
    //         mel.push_back(sign *
    //                       g_(i / 2 * norb_ + j / 2, k / 2 * norb_ + l / 2));
    //       }
    //     }
    //   }
    // }

    // for (int i = 0; i < norbs_; i++) {
    //   if (){continue;}
    //
    //   // Diagonal excitations
    //   mel[0] += h_(i, i);
    //
    //   // Off-diagonal excitations
    //   for (intj = 0; j < norbs_; j++) {
    //     if (h_(i, j) > 0) {
    //       connectors.push_back({i, j});
    //     }
    //   }
    // }

    // for (std::size_t opn = 0; opn < nops_; opn++) {
    //   int st1 = StateNumber(v, opn);
    //
    //   assert(st1 < int(mat_[opn].size()));
    //   assert(st1 < int(connected_[opn].size()));
    //
    //   mel[0] += (mat_[opn][st1][st1]);
    //
    //   // off-diagonal part
    //   for (auto st2 : connected_[opn][st1]) {
    //     connectors.push_back(sites_[opn]);
    //     assert(st2 < int(states_[opn].size()));
    //     newconfs.push_back(states_[opn][st2]);
    //     mel.push_back(mat_[opn][st1][st2]);
    //   }
    // }
  }

  void GetOpenClosed(VectorConstRefType v, std::vector<int> &open,
                     std::vector<int> &closed) const {
    // Find open and closed spin orbitals
    for (int i = 0; i < norb_; i++) {
      if (v(i) == 0) { // unoccupied
        open.push_back(i);
      } else { // occupied
        closed.push_back(i);
      }
    }
  }

  double Parity(std::vector<double> &conf, int i, int j) const {
    // TODO check
    double sign = 1.0;

    // Annihilation operator
    for (int b = 0; b < j; b++) {
      if (conf.at(b) == 1.) {
        sign *= -1.0;
      }
    }

    // Creation operator
    for (int a = 0; a < j; a++) {
      if (conf.at(a) == 1.) {
        sign *= -1.0;
      }
    }

    if (i >= j) {
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
