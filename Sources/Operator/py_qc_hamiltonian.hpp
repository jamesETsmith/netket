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

#ifndef NETKET_PYQCHAMILTONIAN_HPP
#define NETKET_PYQCHAMILTONIAN_HPP
// clang-format off
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include "qc_hamiltonian.hpp"
// clang-format on

namespace py = pybind11;

namespace netket {

void AddQCHamiltonian(py::module &subm) {
  py::class_<QCHamiltonian, AbstractOperator>(
      subm, "QCHamiltonian", R"EOF(A custom local operator.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>,
                    double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("h"),
           py::arg("g"), py::arg("e0"),
           R"EOF(
           Constructs a new ``QCHamiltonian`` given a hilbert space and (if
           specified) a constant level shift.

           Args:
               hilbert: Hilbert space the operator acts on.
               h: one-body quantum chemistry integrals
               g: two-body quantum chemistry integrals

           Examples:
               Constructs a ``QCHamiltonian`` without any operators.


           )EOF");
}

} // namespace netket

#endif
