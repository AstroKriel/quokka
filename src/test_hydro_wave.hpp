#ifndef TEST_HYDRO_WAVE_HPP_ // NOLINT
#define TEST_HYDRO_WAVE_HPP_
//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_hydro_wave.hpp
/// \brief Defines a test problem for a linear hydro wave.
///

// external headers
#include "matplotlibcpp.h"
#include <fmt/format.h>

// internal headers
#include "athena_arrays.hpp"
#include "hydro_system.hpp"

// function definitions
void testproblem_hydro_wave();

#endif // TEST_HYDRO_WAVE_HPP_
