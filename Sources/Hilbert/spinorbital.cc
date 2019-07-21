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

SpinOrbital::SpinOrbital(const AbstractGraph &graph, const int nelec)
    : graph_(graph), nelec_(nelec) {
  size_ = graph.Size();
  norb_ = graph.Size();
  InfoMessage() << "SpinOrbital Hilbert space created" << std::endl;
  InfoMessage() << nelec_ << " electons in " << norb_ << " spin orbitals"
                << std::endl;
}

bool SpinOrbital::IsDiscrete() const { return true; }

int SpinOrbital::LocalSize() const { return norb_; }

int SpinOrbital::Size() const { return size_; }

std::vector<double> SpinOrbital::LocalStates() const { return local_; }

void SpinOrbital::RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                             netket::default_random_engine &rgen) const {
  // Populate the state assuming HF (or HF like orbital ordering)
  // i.e. this is not random

  // std::uniform_int_distribution<int> distribution(0, norb_ - 1);

  assert(state.size() == size_);

  //
  for (int i = 0; i < state.size(); i++) {
    if (i < nelec_) {
      state(i) = local_[1];
    } else {
      state(i) = local_[0];
    }
    // state(i) = local_[distribution(rgen)];
  }
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
}; // namespace netket
