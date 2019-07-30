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
#include <array>
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
    InfoMessage() << "Initializing OneBodyIntegrals\n";
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
    InfoMessage() << "Initializing TwoBodyIntegrals\n";
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

  // For QC only
  using DetVectorType = std::vector<std::array<int, 5>>;

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
  const int sz_;
  const double constant_;

  // Resized vectors for FindConn
  mutable std::vector<int> occ;   // Keep track of occ spin orbitals
  mutable std::vector<int> unocc; // Keep trach of unocc spin orbitals
  mutable DetVectorType dets;

  static constexpr double mel_cutoff_ = 1.0e-8;

public:
  explicit QCHamiltonian(std::shared_ptr<const AbstractHilbert> hilbert,
                         const RowMatrixXd &h, const RowMatrixXd &g, int sz,
                         double e0, int nelec)
      : AbstractOperator(hilbert), h_(h), g_(g), norb_(hilbert->Size()),
        nmo_(hilbert->Size()), nelec_(nelec), sz_(sz), constant_(e0) {
    // Check that dimension of one and two body integrals match dimension of the
    // hilbert space
    if (norb_ != h_.N * 2 || norb_ != g_.N * 2) {
      throw InvalidInputError(
          "Matrix size in operator is inconsistent with Hilbert space");
    }

    // Reserve space for vectors (this is for performance)
    occ.reserve(100);
    unocc.reserve(100);
    dets.reserve(1000);

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

    // Runtime test for correct spin and # electrons
    // return;
    int test_nelec = 0;
    int test_sz = 0;
    for (int i = 0; i < v.size(); i++) {
      if (v(i) == 1) {
        test_nelec += 1; // Check for particle number
        if (i % 2 == 0) {
          test_sz += 1; // Check for conservations of Sz
        } else {
          test_sz -= 1;
        }
      }
    }
    if (test_nelec != nelec_ || test_sz != sz_) {
      // printf("test_nelec = %f\n", test_nelec);
      // printf("test_sz = %f\n", test_sz);
      // printf("State =\n");
      // std::cout << v << "\n";
      return;
    }

    // Clear the reserved data structures
    occ.clear();
    unocc.clear();
    dets.clear();

    connectors.resize(1);
    newconfs.resize(1);
    mel.resize(1);

    // mel[0] = constant_;
    mel[0] = 0;
    connectors[0].resize(0);
    newconfs[0].resize(0);
    GetOpenClosed(v, occ, unocc);

    // Excitation = 0 (no new configurations generated)
    // Create occ/unocc as class members and reset at beginning
    // Change dets to std::vector<std::array<int,5>>
    mel[0] += Calculate_Hij(occ);
    // std::vector<std::array<int, 5>> dets;
    GenSinglyExcitedDets(occ, unocc, dets);
    GenDoublyExcitedDets(occ, unocc, dets);

    for (const auto &det : dets) {
      // Single excitation (one-body operators)
      if (det.at(0) == 2) {
        int p = det[2];
        int q = det[1];
        double Hij = Calculate_Hij(occ, p, q);
        if (std::abs(Hij) < mel_cutoff_) {
          continue;
        }
        connectors.push_back({q, p});
        newconfs.push_back({0., 1.});
        mel.push_back(Hij);
      }
      // Double excitations (two-body operators)
      else if (det.at(0) == 4) {
        int p = det[4];
        int q = det[3];
        int r = det[2];
        int s = det[1];

        double Hij = Calculate_Hij(occ, v, p, q, r, s);
        if (std::abs(Hij) < mel_cutoff_) {
          continue;
        }
        connectors.push_back({s, r, q, p});
        newconfs.push_back({0., 1., 0., 1.});
        mel.push_back(Hij);
      } else {
        printf("Problem determinant ");
        for (auto d : dets) {
          printf("%d ", d);
        }
        std::runtime_error{"Determinant doesn't have length two or four."};
      }
    }

    if (occ.size() != nelec_) { // TODO for debugging only
      throw std::runtime_error{"WE LOST SOME ELECTRONS"};
    }
    // printf("//////////////////////////\n");
    // printf("| ");
    // for (auto o : occ) {
    //   printf("%d ", o);
    // }
    // printf(">\n");
    // printf("Size of mel = %d\n", mel.size());
    // // GenFCIVec();
    // MakeH(GenFCIVec());
    // printf("//////////////////////////\n");

  } // End FindConn

  std::vector<std::vector<int>> GenFCIVec() const {
    std::vector<std::vector<int>> occs;
    // All configs (generated by me)
    occs.push_back({0, 1, 2, 3});
    occs.push_back({0, 1, 2, 5});
    occs.push_back({0, 1, 2, 7});
    occs.push_back({0, 3, 2, 5});
    occs.push_back({0, 3, 2, 7});
    occs.push_back({0, 5, 2, 7});
    occs.push_back({0, 1, 4, 3});
    occs.push_back({0, 1, 4, 5});
    occs.push_back({0, 1, 4, 7});
    occs.push_back({0, 3, 4, 5});
    occs.push_back({0, 3, 4, 7});
    occs.push_back({0, 5, 4, 7});
    occs.push_back({0, 1, 6, 3});
    occs.push_back({0, 1, 6, 5});
    occs.push_back({0, 1, 6, 7});
    occs.push_back({0, 3, 6, 5});
    occs.push_back({0, 3, 6, 7});
    occs.push_back({0, 5, 6, 7});
    occs.push_back({2, 1, 4, 3});
    occs.push_back({2, 1, 4, 5});
    occs.push_back({2, 1, 4, 7});
    occs.push_back({2, 3, 4, 5});
    occs.push_back({2, 3, 4, 7});
    occs.push_back({2, 5, 4, 7});
    occs.push_back({2, 1, 6, 3});
    occs.push_back({2, 1, 6, 5});
    occs.push_back({2, 1, 6, 7});
    occs.push_back({2, 3, 6, 5});
    occs.push_back({2, 3, 6, 7});
    occs.push_back({2, 5, 6, 7});
    occs.push_back({4, 1, 6, 3});
    occs.push_back({4, 1, 6, 5});
    occs.push_back({4, 1, 6, 7});
    occs.push_back({4, 3, 6, 5});
    occs.push_back({4, 3, 6, 7});
    occs.push_back({4, 5, 6, 7});

    // All configs from Dice
    // occs.push_back({0, 1, 2, 3}); // 0
    // occs.push_back({1, 2, 3, 4}); // 1
    // occs.push_back({0, 2, 3, 5}); // 2
    // occs.push_back({0, 1, 4, 5}); // 3
    // occs.push_back({2, 3, 4, 5}); // 4
    // occs.push_back({0, 1, 3, 6}); // 5
    // occs.push_back({1, 3, 4, 6}); // 6
    // occs.push_back({1, 2, 5, 6}); // 7
    // occs.push_back({0, 3, 5, 6}); // 8
    // occs.push_back({0, 1, 2, 7}); // 9
    // occs.push_back({1, 2, 4, 7}); // 10
    // occs.push_back({0, 3, 4, 7}); // 11
    // occs.push_back({0, 2, 5, 7}); // 12
    // occs.push_back({0, 1, 6, 7}); // 13
    // occs.push_back({2, 3, 6, 7}); // 14
    // occs.push_back({3, 4, 5, 6}); // 15
    // occs.push_back({2, 4, 5, 7}); // 16
    // occs.push_back({1, 4, 6, 7}); // 17
    // occs.push_back({0, 5, 6, 7}); // 18
    // occs.push_back({4, 5, 6, 7}); // 19

    // H2
    // occs.push_back({0, 1});
    // occs.push_back({2, 3});
    return occs;
  }

  /**
   * @brief Calculate the orbital indices of the excitation that connects the
   * bra and the ket.
   *
   * @param bra Occ vector for bra.
   * @param ket Occ vector for ket.
   * @return std::vector<int>
   */
  std::vector<int> GetOrbDiff(std::vector<int> &bra,
                              std::vector<int> &ket) const {

    std::vector<int> orbdiff;
    std::vector<int> cre;
    std::vector<int> des;

    // record unique creation operators
    bool in_ket = false;
    for (auto b : bra) {
      for (auto k : ket) {
        if (b == k) {
          in_ket = true;
          break;
        }
      }
      if (!in_ket) {
        cre.push_back(b);
      }
      in_ket = false;
    }

    // record unique destruction operators
    bool in_bra = false;
    for (auto k : ket) {
      for (auto b : bra) {
        if (b == k) {
          in_bra = true;
          break;
        }
      }
      if (!in_bra) {
        des.push_back(k);
      }
      in_bra = false;
    }

    //
    if (des.size() != cre.size()) {
      throw std::runtime_error{
          "Number of creation and annihilation operators doesn't match"};
    }
    // Reading left to right s r^+ q p^+
    for (int i = 0; i < des.size(); i++) {
      orbdiff.push_back(des.at(i));
      orbdiff.push_back(cre.at(i));
    }

    // Checks
    if (orbdiff.size() > ket.size() * 2) {

      printf("Orbdiff ");
      for (auto od : orbdiff) {
        printf("%d ", od);
      }
      printf("\n");
      throw std::runtime_error{"Too many orbitals in orbdiff"};
    }

    return orbdiff;
  }

  void MakeH(std::vector<std::vector<int>> confs) const {
    // Make matrix
    Eigen::MatrixXd H(confs.size(), confs.size());

    // Iterate through and calculate orbdiff
    for (int i = 0; i < confs.size(); i++) {
      auto I = confs.at(i);
      for (int j = 0; j < confs.size(); j++) {
        auto J = confs.at(j);
        auto orbdiff = GetOrbDiff(I, J);
        // Use size() of orbdiff to calculate Hij

        if (orbdiff.size() == 0) {
          H(i, j) = Calculate_Hij(J);
        } else if (orbdiff.size() == 2) {
          H(i, j) = Calculate_Hij(J, orbdiff.at(1), orbdiff.at(0));
        } else if (orbdiff.size() == 4) {
          // VectorConstRefType v(j.begin(), j.end());
          // VectorConstRefType v = Eigen::Map<VectorConstRefType>(J.data());
          Eigen::VectorXd vj(norb_);
          vj *= 0.0;
          for (int Ji = 0; Ji < J.size(); Ji++) {
            vj(J.at(Ji)) = 1.;
          }
          H(i, j) = Calculate_Hij(J, vj, orbdiff.at(3), orbdiff.at(2),
                                  orbdiff.at(1), orbdiff.at(0));
        } else {
          H(i, j) = 0.;
          // throw std::runtime_error{"Size of orbdiff not recognized"};
        }
      }
    }
    // Diagonalize
    // Eigen::EigenSolver<MatrixXd> es(H);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(H);
    printf("Eigenvalues fo H\n");
    std::cout << std::setprecision(14);
    std::cout << H << "\n";
    std::cout << es.eigenvalues() << "\n";
  }

  /**
   * @brief Generate all possible singly excited configurations give a reference
   * configuration (occ, unocc).
   *
   * @param occ The vector of occupied spin orbitals.
   * @param unocc The vector of unoccupied spin orbitals.
   * @param dets A 2D vector, containing the indices of the configuration to
   * update, e.g. s r^+ q p^+.
   */
  void GenSinglyExcitedDets(std::vector<int> &occ, std::vector<int> &unocc,
                            DetVectorType &dets) const {
    for (auto p : unocc) {
      for (auto q : occ) {
        if (p % 2 == q % 2) {
          dets.push_back({2, q, p, 0, 0});
        }
      }
    }
  }

  /**
   * @brief Generate all possible double excited configurations give a reference
   * configuration (occ, unocc).
   *
   * @param occ The vector of occupied spin orbitals.
   * @param unocc The vector of unoccupied spin orbitals.
   * @param dets A 2D vector, containing the indices of the configuration to
   * update, e.g. s r^+ q p^+.
   */
  void GenDoublyExcitedDets(std::vector<int> &occ, std::vector<int> &unocc,
                            DetVectorType &dets) const {
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
            // Adding operators in the order in which they are applied to the
            // ket
            dets.push_back({4, ob, ub, oa, ua});
          }
        }
      }
    }

    // 2)
    for (int i1 = 0; i1 < occAlpha.size(); i1++) {
      for (int i2 = i1 + 1; i2 < occAlpha.size(); i2++) {
        for (int j1 = 0; j1 < unoccAlpha.size(); j1++) {
          for (int j2 = j1 + 1; j2 < unoccAlpha.size(); j2++) {
            int q = occAlpha.at(i1);
            int s = occAlpha.at(i2);
            int p = unoccAlpha.at(j1);
            int r = unoccAlpha.at(j2);
            // Adding operators in the order in which they are applied to the
            // ket
            dets.push_back({4, s, r, q, p});
          }
        }
      }
    }

    // 3)
    for (int i1 = 0; i1 < occBeta.size(); i1++) {
      for (int i2 = i1 + 1; i2 < occBeta.size(); i2++) {
        for (int j1 = 0; j1 < unoccBeta.size(); j1++) {
          for (int j2 = j1 + 1; j2 < unoccBeta.size(); j2++) {
            int q = occBeta.at(i1);
            int s = occBeta.at(i2);
            int p = unoccBeta.at(j1);
            int r = unoccBeta.at(j2);
            // Adding operators in the order in which they are applied to the
            // ket
            dets.push_back({4, s, r, q, p});
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

    for (int ii = 0; ii < occ.size(); ii++) {
      int i = occ.at(ii);
      Hij += h_(i, i);

      for (int ji = ii + 1; ji < occ.size(); ji++) {
        int j = occ.at(ji);
        // Direct
        Hij += g_(i, i, j, j);
        if (i % 2 == j % 2) {
          // Exchange
          Hij -= g_(i, j, j, i);
        }
      }
    }

    return Hij;
  }

  /**
   * @brief Returns the Hamiltonian matrix element between two configurations
   * that differ by a single excitation operator (one creation and one
   * excitation operator). This is Case 2 in Tables 2.3/2.4 in Szabo and
   * Ostlund (1989).
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
    // printf("Occ %d %d %d %d p=%d q=%d\tHij = %f\n", occ[0], occ[1], occ[2],
    //        occ[3], p, q, par * Hij);

    return Hij * par;
  }

  /**
   * @brief Returns the Hamiltonian matrix element between two configurations
   * that differ by a double excitation operator (one creation and one
   * excitation operator). This is Case 3 in Tables 2.3/2.4 in Szabo and
   * Ostlund (1989).
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

    if (p % 2 == q % 2) {
      Hij += g_(p, q, r, s);
    }

    if (q % 2 == r % 2) {
      Hij -= g_(p, s, r, q);
    }

    Hij *= par;
    return Hij;
  }

  void GetOpenClosed(VectorConstRefType v, std::vector<int> &occ,
                     std::vector<int> &unocc) const {
    // Find occ and unocc spin orbitals
    for (int i = 0; i < v.size(); i++) {
      if (v(i) == 0) { // unoccupied
        unocc.push_back(i);
      } else if (v(i) == 1) { // occupied
        occ.push_back(i);
      } else {
        std::cout << v << std::endl;
        throw std::runtime_error{
            "Occupation number not supported for QCHamiltonian\n"};
      }
    }
  }

  double Parity(VectorConstRefType v, int p, int q) const {
    // p^+ q
    // q -> p
    double sign = 1.0;

    for (int i = 0; i < q; i++) {
      if (v(i) == 1) {
        sign *= -1.;
      }
    }

    for (int i = 0; i < p; i++) {
      if (v(i) == 1) {
        sign *= -1.;
      }
    }

    if (p > q) {
      sign *= -1.;
    }
    return sign;
  }

  double Parity(VectorConstRefType v, int p, int q, int r, int s) const {
    // p^+ r^+ s q
    // q -> p
    // s -> r
    double sign = 1.0;
    sign *= Parity(v, p, q);

    Eigen::VectorXd v2 = v;

    // Change occupation
    v2(p) = 1;
    v2(q) = 0;

    sign *= Parity(v2, r, s);

    // Reset occupation
    v2(p) = 0;
    v2(q) = 1;

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
