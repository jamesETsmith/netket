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

#include "spinorbital.hpp"

namespace netket {

SpinOrbital::SpinOrbital(const AbstractGraph &graph, const int nelec,
                         const int sz)
    : graph_(graph), nelec_(nelec), sz_(sz) {
  size_ = graph.Size();
  norb_ = graph.Size();

  // Reserve the space for alpha and beta vectors
  alpha.reserve(100);
  beta.reserve(100);

  InfoMessage() << "SpinOrbital Hilbert space created" << std::endl;
  InfoMessage() << nelec_ << " electons in " << norb_ << " spin orbitals"
                << std::endl;
}

bool SpinOrbital::IsDiscrete() const { return true; }

int SpinOrbital::LocalSize() const { return local_.size(); }

int SpinOrbital::Size() const { return size_; }

std::vector<double> SpinOrbital::LocalStates() const { return local_; }

void SpinOrbital::RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                             netket::default_random_engine &rgen) const {
  assert(state.size() == size_);

  // Keep track of a few helpful numbers
  int nalpha = nelec_ / 2 + sz_;
  int nbeta = nelec_ / 2 - sz_;
  int nspatialorbs = state.size() / 2;

  // Reset alpha and beta
  alpha.clear();
  beta.clear();

  // Populate alpha and beta with all possible spinorbital indices
  // for (int i = 0; i < nspatialorbs; i++) {
  //   alpha.push_back(i);
  //   beta.push_back(i);
  //   state(2 * i) = 0.0;
  //   state(2 * i + 1) = 0.0;
  // }

  // // Shuffle them and then select the first nalpha and nbeta
  // std::shuffle(alpha.begin(), alpha.end(), rgen);
  // std::shuffle(beta.begin(), beta.end(), rgen);

  // for (int a = 0; a < nalpha; a++) {
  //   state(2 * alpha.at(a)) = 1.;
  // }

  // for (int b = 0; b < nbeta; b++) {
  //   state(2 * beta.at(b) + 1) = 1.;
  // }

  // Non-random configuration (HF)
  for (int i = 0; i < nspatialorbs; i++) {
    state(2 * i) = local_[i < nalpha]; // 1 if i < nalpha else 0
    state(2 * i + 1) = local_[i < nbeta];
  }

  // for (int i = 0; i < state.size(); i++) {
  //   if (i < nelec_) {
  //     state(i) = local_[1];
  //   } else {
  //     state(i) = local_[0];
  //   }
  //   // state(i) = local_[distribution(rgen)];
  // }
}

void SpinOrbital::UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                             const std::vector<int> &tochange,
                             const std::vector<double> &newconf) const {
  assert(v.size() == size_);

  int i = 0;
  for (auto sf : tochange) {
    v(sf) = newconf[i];
    i++;
  }
} // namespace netket

const AbstractGraph &SpinOrbital::GetGraph() const noexcept { return graph_; }
} // namespace netket
