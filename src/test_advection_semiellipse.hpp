#ifndef TEST_ADVECTION_HPP_ // NOLINT
#define TEST_ADVECTION_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>

// internal headers
#include "athena_arrays.hpp"
#include "linear_advection.hpp"

// function definitions
template <typename problem_t> void write_density(LinearAdvectionSystem<problem_t> &advection_system);
int testproblem_advection();

#endif // TEST_ADVECTION_HPP_