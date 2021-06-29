#ifndef TEST_RADIATION_SHADOW_HPP_ // NOLINT
#define TEST_RADIATION_SHADOW_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_radiation_shadow.hpp
/// \brief Defines a test problem for radiation in the static diffusion regime.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>
#include <fstream>

// internal headers

#include "radiation_system.hpp"
#include "RadhydroSimulation.hpp"

// function definitions
auto testproblem_radiation_shadow() -> int;

#endif // TEST_RADIATION_SHADOW_HPP_