//==============================================================================
// Copyright 2026 Nicholas Kriel.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testRiemannSignSweep.cpp
/// \brief Sweeps HLLC across left/right states and carbuncle-sensor
/// diagnostics, checking the mass-flux sign against the exact isothermal
/// Riemann solution.

#include <algorithm>
#include <array>
#include <cmath>

#include "AMReX.H"
#include "AMReX_GpuAsyncArray.H"
#include "AMReX_GpuLaunch.H"
#include "AMReX_Print.H"

#include "hydro/HLLC.hpp"
#include "physics_info.hpp"

struct RiemannSignSweepProblem {};

template <> struct quokka::EOS_Traits<RiemannSignSweepProblem> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<RiemannSignSweepProblem> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
};

namespace
{
// Isothermal Riemann problem wave functions [Toro (2009), Ch. 4, specialised
// to constant sound speed]: shock branch for rho_star >= rho_K, rarefaction
// branch otherwise. Returns the magnitude of the velocity change from state K
// to the star region.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto isothermalWaveFunction(double rho_star, double rho_K, double cs) -> double
{
	if (rho_star >= rho_K) {
		return (rho_star - rho_K) * cs / std::sqrt(rho_K * rho_star);
	}
	return cs * std::log(rho_star / rho_K);
}

// Exact isothermal Riemann solution: returns the sign of the contact-wave
// velocity u_star, which equals the sign of the physically correct mass
// flux at the interface. Both wave functions are monotonic in rho_star, so
// bisection over a generous bracket converges unconditionally; only the
// sign is needed here, not the full solution.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto exactIsothermalMassFluxSign(double rho_L, double u_L, double rho_R, double u_R, double cs) -> double
{
	const double velocity_jump = u_L - u_R;
	double rho_lo = 1.0e-12 * std::min(rho_L, rho_R);
	double rho_hi = 1.0e12 * std::max(rho_L, rho_R);

	for (int iteration = 0; iteration < 100; ++iteration) {
		const double rho_mid = 0.5 * (rho_lo + rho_hi);
		const double residual = isothermalWaveFunction(rho_mid, rho_L, cs) + isothermalWaveFunction(rho_mid, rho_R, cs) - velocity_jump;
		if (residual > 0.0) {
			rho_hi = rho_mid;
		} else {
			rho_lo = rho_mid;
		}
	}

	const double rho_star = 0.5 * (rho_lo + rho_hi);
	const double u_star = u_L - isothermalWaveFunction(rho_star, rho_L, cs);
	if (u_star > 0.0) {
		return 1.0;
	}
	if (u_star < 0.0) {
		return -1.0;
	}
	return 0.0;
}
} // namespace

auto problem_main() -> int
{
	constexpr double cs = quokka::EOS_Traits<RiemannSignSweepProblem>::cs_isothermal;
	constexpr int density_ratio_count = 5;
	constexpr int velocity_jump_count = 5;
	constexpr int transverse_jump_count = 7;
	constexpr int num_cases = density_ratio_count * velocity_jump_count * transverse_jump_count;

	std::array<double, num_cases> host_flux{};
	amrex::AsyncArray async_flux(host_flux.data(), num_cases);
	double *const flux_out = async_flux.data();

	amrex::ParallelFor(num_cases, [=] AMREX_GPU_DEVICE(int case_index) noexcept {
		const int density_ratio_index = case_index % density_ratio_count;
		const int velocity_jump_index = (case_index / density_ratio_count) % velocity_jump_count;
		const int transverse_jump_index = case_index / (density_ratio_count * velocity_jump_count);

		// density ratio spans two orders of magnitude either side of unity
		const double density_ratio_exponent = -2.0 + 4.0 * density_ratio_index / (density_ratio_count - 1);
		const double density_ratio = std::pow(10.0, density_ratio_exponent);
		const double rho_L = 1.0;
		const double rho_R = rho_L * density_ratio;

		// normal velocity jump spans weak to strongly supersonic compression
		const double velocity_jump_magnitude = cs * (0.1 + 4.0 * velocity_jump_index / (velocity_jump_count - 1));
		const double u_L = 0.5 * velocity_jump_magnitude;
		const double u_R = -0.5 * velocity_jump_magnitude;

		// transverse jump spans mild to extreme relative to the normal jump,
		// independent of the physical state, matching how #2313's dw arose
		// from unrelated transverse compression
		const double dw_magnitude = cs * (0.1 + 20.0 * transverse_jump_index / (transverse_jump_count - 1));
		const double dw = -dw_magnitude;
		const double du = u_R - u_L;

		quokka::HydroState<0, 0> left{};
		quokka::HydroState<0, 0> right{};
		left.rho = rho_L;
		left.u = u_L;
		left.P = cs * cs * rho_L;
		left.cs = cs;
		right.rho = rho_R;
		right.u = u_R;
		right.P = cs * cs * rho_R;
		right.cs = cs;

		auto const flux = quokka::Riemann::HLLC<RiemannSignSweepProblem, 0, 0, 6>(left, right, quokka::EOS_Traits<RiemannSignSweepProblem>::gamma, du, dw);
		const double expected_sign = exactIsothermalMassFluxSign(rho_L, u_L, rho_R, u_R, cs);
		flux_out[case_index] = (expected_sign == 0.0) ? 0.0 : flux[0] * expected_sign;
	});
	async_flux.copyToHost(host_flux.data(), num_cases);

	bool all_ok = true;
	int mismatch_count = 0;
	for (int case_index = 0; case_index < num_cases; ++case_index) {
		if (host_flux[case_index] < 0.0) {
			all_ok = false;
			++mismatch_count;
		}
	}
	amrex::Print() << mismatch_count << " of " << num_cases << " cases disagree in sign with the exact isothermal solution\n";
	amrex::Print() << (all_ok ? "test passed\n" : "test failed\n");
	return all_ok ? 0 : 1;
}
