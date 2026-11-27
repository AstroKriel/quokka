// SPDX-FileCopyrightText: © The Quokka Authors
// SPDX-License-Identifier: MIT

/// \file testMHDSmallScaleDynamo.cpp
/// \brief Small Scale Dynamo simulation.
///

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "hydro/hydro_system.hpp"
#include "hydro/mhd_system.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_REAL.H"
#include "AMReX_Random.H"
#ifdef AMREX_USE_FFT
#include "AMReX_FFT.H"
#endif
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

struct MHDSmallScaleDynamo {};

// isothermal EOS: pressure = cs^2 * rho (no thermal energy equation)
template <> struct quokka::EOS_Traits<MHDSmallScaleDynamo> {
	static constexpr double gamma = 1.0;
	static constexpr double cs_isothermal = 1.0; // dimensionless sound speed
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDSmallScaleDynamo> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true; // solve the Euler equations
	static constexpr bool is_mhd_enabled = true;   // solve the MHD equations
	static constexpr int nDustGroups = 0;	       // no dust groups
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr amrex::Real gravitational_constant = 1.0;
	static constexpr ResistivityModel resistivity_model = ResistivityModel::constant; // eta from mhd.resistivity in the TOML; defaults to 0
	static constexpr ViscosityModel viscosity_model = ViscosityModel::constant;	  // shear/bulk viscosity from the TOML; both default to 0
};

// isothermal EOS has no internal energy to reconstruct; pressure computed directly from rho
template <> struct HydroSystem_Traits<MHDSmallScaleDynamo> {
	static constexpr bool reconstruct_eint = false;
};

// uniform density, zero velocity; turbulent driving (configured in input file) stirs the gas
template <> void QuokkaSimulation<MHDSmallScaleDynamo>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::density_index) = 1.0;
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::energy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDSmallScaleDynamo>::internalEnergy_index) = 0.0;
	});
}

int seed_b_wavenumber = 2;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
double seed_b_fraction = 0.0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
double target_vdisp = 0.0;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// rms of the perturbation field, as a fraction of the rms of the ABC seed field
double seed_b_perturbation_fraction = 0.0; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

struct VectorPotentialPerturbationMode {
	amrex::GpuArray<double, 3> wavevector;
	amrex::GpuArray<double, 3> amplitude;
	double phase;
};

constexpr int max_perturbation_modes = 32;
constexpr unsigned int perturbation_rng_seed = 42;
amrex::GpuArray<VectorPotentialPerturbationMode, max_perturbation_modes> perturbation_modes{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
int num_perturbation_modes = 0;								       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// random transverse plane waves in the vector potential, at integer wavevectors with 0 < |k| < seed_b_wavenumber
void initializePerturbationModes(const double box_length)
{
	if (seed_b_perturbation_fraction <= 0.0) {
		return;
	}
	const int max_component = seed_b_wavenumber - 1;
	const int max_squared_wavenumber = seed_b_wavenumber * seed_b_wavenumber;
	std::vector<amrex::GpuArray<int, 3>> candidate_wavevectors;
	for (int kx = 0; kx <= max_component; ++kx) {
		for (int ky = -max_component; ky <= max_component; ++ky) {
			for (int kz = -max_component; kz <= max_component; ++kz) {
				const int squared_wavenumber = (kx * kx) + (ky * ky) + (kz * kz);
				const bool wavevector_is_in_half_space = (kx > 0) || ((kx == 0) && (ky > 0)) || ((kx == 0) && (ky == 0) && (kz > 0));
				if (wavevector_is_in_half_space && (squared_wavenumber < max_squared_wavenumber)) {
					candidate_wavevectors.push_back({kx, ky, kz});
				}
			}
		}
	}
	std::mt19937 generator(perturbation_rng_seed);
	const auto draw_uniform = [&generator]() -> double { return static_cast<double>(generator()) / 4294967296.0; };
	const std::size_t num_candidates = candidate_wavevectors.size();
	num_perturbation_modes = static_cast<int>(std::min(num_candidates, static_cast<std::size_t>(max_perturbation_modes)));
	const double perturbation_b_rms = seed_b_perturbation_fraction * std::sqrt(seed_b_fraction) * target_vdisp;
	const double mode_b_amplitude = perturbation_b_rms * std::sqrt(2.0 / num_perturbation_modes);
	for (int mode_index = 0; mode_index < num_perturbation_modes; ++mode_index) {
		const std::size_t mode_position = static_cast<std::size_t>(mode_index);
		const std::size_t swap_position = mode_position + (generator() % (num_candidates - mode_position));
		std::swap(candidate_wavevectors[mode_position], candidate_wavevectors[swap_position]);
		const amrex::GpuArray<int, 3> integer_wavevector = candidate_wavevectors[mode_position];
		const double wavenumber_scale = 2.0 * M_PI / box_length;
		const amrex::GpuArray<double, 3> wavevector = {wavenumber_scale * integer_wavevector[0], wavenumber_scale * integer_wavevector[1],
							       wavenumber_scale * integer_wavevector[2]};
		const double wavenumber = std::sqrt((wavevector[0] * wavevector[0]) + (wavevector[1] * wavevector[1]) + (wavevector[2] * wavevector[2]));
		amrex::GpuArray<double, 3> transverse_vector = {0.0, 0.0, 0.0};
		double transverse_norm = 0.0;
		while (transverse_norm < 1.0e-6 * wavenumber) {
			const amrex::GpuArray<double, 3> random_vector = {(2.0 * draw_uniform()) - 1.0, (2.0 * draw_uniform()) - 1.0,
									  (2.0 * draw_uniform()) - 1.0};
			transverse_vector = {(wavevector[1] * random_vector[2]) - (wavevector[2] * random_vector[1]),
					     (wavevector[2] * random_vector[0]) - (wavevector[0] * random_vector[2]),
					     (wavevector[0] * random_vector[1]) - (wavevector[1] * random_vector[0])};
			transverse_norm = std::sqrt((transverse_vector[0] * transverse_vector[0]) + (transverse_vector[1] * transverse_vector[1]) +
						    (transverse_vector[2] * transverse_vector[2]));
		}
		const double vector_potential_amplitude = mode_b_amplitude / wavenumber;
		perturbation_modes[mode_index].wavevector = wavevector;
		perturbation_modes[mode_index].amplitude = {vector_potential_amplitude * transverse_vector[0] / transverse_norm,
							    vector_potential_amplitude * transverse_vector[1] / transverse_norm,
							    vector_potential_amplitude * transverse_vector[2] / transverse_norm};
		perturbation_modes[mode_index].phase = 2.0 * M_PI * draw_uniform();
	}
}

// evaluates one vector potential component at the centre of the cell edge aligned with that axis
[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto
computePerturbationVectorPotential(amrex::GpuArray<VectorPotentialPerturbationMode, max_perturbation_modes> const &modes, const int num_modes, const int axis,
				   const int edge_i, const int edge_j, const int edge_k, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
				   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> double
{
	const double x_edge = prob_lo[0] + (edge_i + (0.5 * static_cast<double>(axis == 0))) * dx[0];
	const double y_edge = prob_lo[1] + (edge_j + (0.5 * static_cast<double>(axis == 1))) * dx[1];
	const double z_edge = prob_lo[2] + (edge_k + (0.5 * static_cast<double>(axis == 2))) * dx[2];
	double vector_potential_value = 0.0;
	for (int mode_index = 0; mode_index < num_modes; ++mode_index) {
		const VectorPotentialPerturbationMode &mode = modes[mode_index];
		const double wave_phase = (mode.wavevector[0] * x_edge) + (mode.wavevector[1] * y_edge) + (mode.wavevector[2] * z_edge) + mode.phase;
		vector_potential_value += mode.amplitude[axis] * std::cos(wave_phase);
	}
	return vector_potential_value;
}

// parameters of the Pencil Code initial condition power_randomphase_hel, with the Pencil defaults; the exponents set the
// shell-integrated spectrum of the vector potential, so the magnetic spectrum goes as k^(initpower + 2) below kpeak and
// k^(initpower2 + 2) above it; kpeak = 0 selects the ABC seed
struct PowerRandomphaseHelParams {
	double kpeak = 0.0;
	double initpower = 0.0;
	double initpower2 = -11.0 / 3.0;
	double nfact = 4.0;
	double cutoff = 0.0;
	double ncutoff = 1.0;
	double relhel = 1.0;
	int random_amplitude = 0;
	// 1: A = A_z(x, y) z_hat from kz = 0 modes only, so B lies in the x-y plane and does not depend on z (no helicity)
	int planar_field = 0;
};

PowerRandomphaseHelParams helical_seed_params{};			     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::unique_ptr<amrex::MultiFab> helical_seed_vector_potential_mf = nullptr; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// mode amplitude of the vector potential: (k/kpeak)^(2 mhalf) / [1 + (k/kpeak)^(2 nexp1)]^nexp2, times an optional cutoff
[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeModeAmplitude_PowerRandomphaseHel(PowerRandomphaseHelParams const &params,
												const double squared_wavenumber) -> double
{
	const double scaled_squared_wavenumber = squared_wavenumber / (params.kpeak * params.kpeak);
	const double mhalf = 0.25 * (params.initpower - 2.0);
	const double nexp1 = 0.25 * params.nfact * (params.initpower - params.initpower2);
	const double nexp2 = 1.0 / params.nfact;
	const double broken_power_law = std::pow(scaled_squared_wavenumber, mhalf) / std::pow(1.0 + std::pow(scaled_squared_wavenumber, nexp1), nexp2);
	if (params.cutoff > 0.0) {
		return broken_power_law * std::exp(-0.5 * std::pow(squared_wavenumber / (params.cutoff * params.cutoff), params.ncutoff));
	}
	return broken_power_law;
}

// component axis of the discrete curl, at the face with the same index as cell (i, j, k)
[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeDiscreteCurl(amrex::Array4<const amrex::Real> const &vector_potential_ec_a4, const int axis,
									   const int i, const int j, const int k,
									   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx) -> double
{
	if (axis == 0) {
		return ((vector_potential_ec_a4(i, j + 1, k, 2) - vector_potential_ec_a4(i, j, k, 2)) / dx[1]) -
		       ((vector_potential_ec_a4(i, j, k + 1, 1) - vector_potential_ec_a4(i, j, k, 1)) / dx[2]);
	}
	if (axis == 1) {
		return ((vector_potential_ec_a4(i, j, k + 1, 0) - vector_potential_ec_a4(i, j, k, 0)) / dx[2]) -
		       ((vector_potential_ec_a4(i + 1, j, k, 2) - vector_potential_ec_a4(i, j, k, 2)) / dx[0]);
	}
	return ((vector_potential_ec_a4(i + 1, j, k, 1) - vector_potential_ec_a4(i, j, k, 1)) / dx[0]) -
	       ((vector_potential_ec_a4(i, j + 1, k, 0) - vector_potential_ec_a4(i, j, k, 0)) / dx[1]);
}

[[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto computeSignedWavenumber(const int index, const int num_cells) -> int
{
	if ((2 * index) > num_cells) {
		return index - num_cells;
	}
	return index;
}

#ifdef AMREX_USE_FFT
// edge-centred vector potential from the Pencil Code initial condition power_randomphase_hel: random phases, a broken power
// law spectrum, transverse projection, then the helical operator (delta_ij + i relhel epsilon_ijk khat_k)
auto makeVectorPotential_PowerRandomphaseHel(amrex::Geometry const &geometry, amrex::BoxArray const &box_array,
					     amrex::DistributionMapping const &distribution_map) -> std::unique_ptr<amrex::MultiFab>
{
	using SpectralMultiFab = amrex::FabArray<amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>;
	const amrex::Box domain = geometry.Domain();
	const amrex::GpuArray<int, 3> num_cells = {domain.length(0), domain.length(1), domain.length(2)};
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geometry.CellSizeArray();
	const PowerRandomphaseHelParams params = helical_seed_params;
	// wavenumbers are measured in units of 2 pi / L_x, so integer index n_axis along an axis of length L_axis is n_axis L_x / L_axis
	const amrex::GpuArray<double, 3> wavenumber_unit_ratio = {1.0, geometry.ProbLength(0) / geometry.ProbLength(1),
								  geometry.ProbLength(0) / geometry.ProbLength(2)};
	for (int axis = 0; axis < 3; ++axis) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(2.0 * params.kpeak < num_cells[axis] * wavenumber_unit_ratio[axis],
						 "setup.helical_seed_kpeak must lie below the Nyquist wavenumber of every axis");
	}
	amrex::MultiFab noise(box_array, distribution_map, 3, 0);
	for (amrex::MFIter iter(noise); iter.isValid(); ++iter) {
		const amrex::Array4<amrex::Real> noise_a4 = noise.array(iter);
		amrex::ParallelForRNG(iter.validbox(), 3, [=] AMREX_GPU_DEVICE(int i, int j, int k, int component, amrex::RandomEngine const &engine) {
			noise_a4(i, j, k, component) = amrex::RandomNormal(0.0, 1.0, engine);
		});
	}
	amrex::Gpu::streamSynchronize();
	amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::both> transform(domain);
	const auto [spectral_box_array, spectral_distribution_map] = transform.getSpectralDataLayout();
	SpectralMultiFab noise_spectrum(spectral_box_array, spectral_distribution_map, 3, 0);
	SpectralMultiFab vector_potential_spectrum(spectral_box_array, spectral_distribution_map, 3, 0);
	for (int component = 0; component < 3; ++component) {
		transform.forward(noise, noise_spectrum, component, component);
	}
	for (amrex::MFIter iter(noise_spectrum); iter.isValid(); ++iter) {
		const auto noise_a4 = noise_spectrum.const_array(iter);
		const auto vector_potential_a4 = vector_potential_spectrum.array(iter);
		amrex::ParallelFor(iter.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const amrex::GpuArray<int, 3> integer_wavevector = {computeSignedWavenumber(i, num_cells[0]), computeSignedWavenumber(j, num_cells[1]),
									    computeSignedWavenumber(k, num_cells[2])};
			double squared_wavenumber = 0.0;
			for (int axis = 0; axis < 3; ++axis) {
				const double wavenumber_component = integer_wavevector[axis] * wavenumber_unit_ratio[axis];
				squared_wavenumber += wavenumber_component * wavenumber_component;
			}
			const bool mode_is_nyquist = ((2 * std::abs(integer_wavevector[0])) == num_cells[0]) ||
						     ((2 * std::abs(integer_wavevector[1])) == num_cells[1]) ||
						     ((2 * std::abs(integer_wavevector[2])) == num_cells[2]);
			const bool mode_is_out_of_plane = (params.planar_field == 1) && (integer_wavevector[2] != 0);
			if ((squared_wavenumber == 0.0) || mode_is_nyquist || mode_is_out_of_plane) {
				for (int component = 0; component < 3; ++component) {
					vector_potential_a4(i, j, k, component) = amrex::GpuComplex<amrex::Real>(0.0, 0.0);
				}
				return;
			}
			amrex::GpuArray<double, 3> discrete_wavevector{};
			amrex::GpuArray<amrex::GpuComplex<amrex::Real>, 3> half_cell_shift{};
			for (int axis = 0; axis < 3; ++axis) {
				const double half_cell_phase = M_PI * integer_wavevector[axis] / num_cells[axis];
				discrete_wavevector[axis] = 2.0 * std::sin(half_cell_phase) / dx[axis];
				half_cell_shift[axis] = amrex::GpuComplex<amrex::Real>(std::cos(half_cell_phase), std::sin(half_cell_phase));
			}
			const double discrete_norm =
			    std::sqrt((discrete_wavevector[0] * discrete_wavevector[0]) + (discrete_wavevector[1] * discrete_wavevector[1]) +
				      (discrete_wavevector[2] * discrete_wavevector[2]));
			const amrex::GpuArray<double, 3> unit_wavevector = {discrete_wavevector[0] / discrete_norm, discrete_wavevector[1] / discrete_norm,
									    discrete_wavevector[2] / discrete_norm};
			amrex::GpuArray<amrex::GpuComplex<amrex::Real>, 3> noise_mode = {noise_a4(i, j, k, 0), noise_a4(i, j, k, 1), noise_a4(i, j, k, 2)};
			if (params.random_amplitude == 0) {
				for (int component = 0; component < 3; ++component) {
					noise_mode[component] = noise_mode[component] / amrex::abs(noise_mode[component]);
				}
			}
			const amrex::GpuComplex<amrex::Real> longitudinal =
			    (noise_mode[0] * unit_wavevector[0]) + (noise_mode[1] * unit_wavevector[1]) + (noise_mode[2] * unit_wavevector[2]);
			const amrex::GpuArray<amrex::GpuComplex<amrex::Real>, 3> transverse = {noise_mode[0] - (longitudinal * unit_wavevector[0]),
											       noise_mode[1] - (longitudinal * unit_wavevector[1]),
											       noise_mode[2] - (longitudinal * unit_wavevector[2])};
			const amrex::GpuArray<amrex::GpuComplex<amrex::Real>, 3> rotated = {
			    (transverse[2] * unit_wavevector[1]) - (transverse[1] * unit_wavevector[2]),
			    (transverse[0] * unit_wavevector[2]) - (transverse[2] * unit_wavevector[0]),
			    (transverse[1] * unit_wavevector[0]) - (transverse[0] * unit_wavevector[1])};
			const double amplitude = computeModeAmplitude_PowerRandomphaseHel(params, squared_wavenumber);
			if (params.planar_field == 1) {
				vector_potential_a4(i, j, k, 0) = amrex::GpuComplex<amrex::Real>(0.0, 0.0);
				vector_potential_a4(i, j, k, 1) = amrex::GpuComplex<amrex::Real>(0.0, 0.0);
				vector_potential_a4(i, j, k, 2) = noise_mode[2] * half_cell_shift[2] * amplitude;
				return;
			}
			const amrex::GpuComplex<amrex::Real> helical_factor(0.0, params.relhel);
			for (int component = 0; component < 3; ++component) {
				vector_potential_a4(i, j, k, component) =
				    (transverse[component] + (helical_factor * rotated[component])) * half_cell_shift[component] * amplitude;
			}
		});
	}
	amrex::Gpu::streamSynchronize();
	auto vector_potential_ec_mf = std::make_unique<amrex::MultiFab>(box_array, distribution_map, 3, 2);
	for (int component = 0; component < 3; ++component) {
		transform.backward(vector_potential_spectrum, *vector_potential_ec_mf, component, component);
	}
	vector_potential_ec_mf->FillBoundary(geometry.periodicity());
	const auto vector_potential_ec_a4s = vector_potential_ec_mf->const_arrays();
	amrex::Real summed_b_squared =
	    amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum>{}, amrex::TypeList<amrex::Real>{}, *vector_potential_ec_mf, amrex::IntVect(0),
			     [=] AMREX_GPU_DEVICE(int box_index, int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
				     const auto vector_potential_ec_a4 = vector_potential_ec_a4s[box_index];
				     const double b_wcomp0 = computeDiscreteCurl(vector_potential_ec_a4, 0, i, j, k, dx);
				     const double b_wcomp1 = computeDiscreteCurl(vector_potential_ec_a4, 1, i, j, k, dx);
				     const double b_wcomp2 = computeDiscreteCurl(vector_potential_ec_a4, 2, i, j, k, dx);
				     return {(b_wcomp0 * b_wcomp0) + (b_wcomp1 * b_wcomp1) + (b_wcomp2 * b_wcomp2)};
			     });
	amrex::ParallelDescriptor::ReduceRealSum(summed_b_squared);
	const double measured_b_rms = std::sqrt(summed_b_squared / static_cast<double>(domain.numPts()));
	const double target_b_rms = std::sqrt(seed_b_fraction) * target_vdisp;
	vector_potential_ec_mf->mult(target_b_rms / measured_b_rms, 0, 3, 2);
	return vector_potential_ec_mf;
}
#endif

template <> void QuokkaSimulation<MHDSmallScaleDynamo>::preCalculateInitialConditions()
{
	if (helical_seed_params.kpeak <= 0.0) {
		return;
	}
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0, "setup.helical_seed_kpeak supports a single AMR level");
#ifdef AMREX_USE_FFT
	helical_seed_vector_potential_mf = makeVectorPotential_PowerRandomphaseHel(geom[0], state_new_cc_[0].boxArray(), state_new_cc_[0].DistributionMap());
#else
	amrex::Abort("setup.helical_seed_kpeak needs a build configured with -DAMReX_FFT=ON");
#endif
}

// we initialise the magnetic field using an ABC (Arnold-Beltrami-Childress) seed field:
// a force-free, maximally helical field defined as the (discrete) curl of a sinusoidal vector potential.
// the general ABC field has three independent amplitudes (per spatial axis); here we choose them to be equal.
template <> void QuokkaSimulation<MHDSmallScaleDynamo>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;

	const double box_length = prob_hi[0] - prob_lo[0];
	const double seed_wavenumber = 2.0 * M_PI * seed_b_wavenumber / box_length;
	const double seed_vecpot_amplitude = std::sqrt(seed_b_fraction / 3.0) * target_vdisp / seed_wavenumber;

	constexpr int b_index = Physics_Indices<MHDSmallScaleDynamo>::mhdFirstIndex;
	if (helical_seed_vector_potential_mf != nullptr) {
		const amrex::Box cell_box = amrex::enclosedCells(indexRange);
		const int axis = static_cast<int>(dir);
		for (const int box_index : helical_seed_vector_potential_mf->IndexArray()) {
			if (helical_seed_vector_potential_mf->boxArray()[box_index] == cell_box) {
				const amrex::Array4<const amrex::Real> vector_potential_ec_a4 = helical_seed_vector_potential_mf->const_array(box_index);
				amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					state_fc(i, j, k, b_index) = computeDiscreteCurl(vector_potential_ec_a4, axis, i, j, k, dx);
				});
			}
		}
		return;
	}
	const amrex::GpuArray<VectorPotentialPerturbationMode, max_perturbation_modes> modes = perturbation_modes;
	const int num_modes = num_perturbation_modes;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const auto perturbation = [&](const int axis, const int edge_i, const int edge_j, const int edge_k) -> double {
			return computePerturbationVectorPotential(modes, num_modes, axis, edge_i, edge_j, edge_k, prob_lo, dx);
		};
		if (dir == quokka::direction::x) {
			// x-face: B_x = dA_z/dy - dA_y/dz
			const double x_face = prob_lo[0] + i * dx[0];
			const double y_hi = prob_lo[1] + (j + 1) * dx[1];
			const double y_lo = prob_lo[1] + j * dx[1];
			const double z_hi = prob_lo[2] + (k + 1) * dx[2];
			const double z_lo = prob_lo[2] + k * dx[2];
			const double vecpot_z_at_y_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * y_hi) + std::cos(seed_wavenumber * x_face));
			const double vecpot_z_at_y_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * y_lo) + std::cos(seed_wavenumber * x_face));
			const double vecpot_y_at_z_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * x_face) + std::cos(seed_wavenumber * z_hi));
			const double vecpot_y_at_z_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * x_face) + std::cos(seed_wavenumber * z_lo));
			state_fc(i, j, k, b_index) = (vecpot_z_at_y_hi - vecpot_z_at_y_lo) / dx[1] - (vecpot_y_at_z_hi - vecpot_y_at_z_lo) / dx[2] +
						     (perturbation(2, i, j + 1, k) - perturbation(2, i, j, k)) / dx[1] -
						     (perturbation(1, i, j, k + 1) - perturbation(1, i, j, k)) / dx[2];
		} else if (dir == quokka::direction::y) {
			// y-face: B_y = dA_x/dz - dA_z/dx
			const double y_face = prob_lo[1] + j * dx[1];
			const double z_hi = prob_lo[2] + (k + 1) * dx[2];
			const double z_lo = prob_lo[2] + k * dx[2];
			const double x_hi = prob_lo[0] + (i + 1) * dx[0];
			const double x_lo = prob_lo[0] + i * dx[0];
			const double vecpot_x_at_z_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * z_hi) + std::cos(seed_wavenumber * y_face));
			const double vecpot_x_at_z_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * z_lo) + std::cos(seed_wavenumber * y_face));
			const double vecpot_z_at_x_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * y_face) + std::cos(seed_wavenumber * x_hi));
			const double vecpot_z_at_x_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * y_face) + std::cos(seed_wavenumber * x_lo));
			state_fc(i, j, k, b_index) = (vecpot_x_at_z_hi - vecpot_x_at_z_lo) / dx[2] - (vecpot_z_at_x_hi - vecpot_z_at_x_lo) / dx[0] +
						     (perturbation(0, i, j, k + 1) - perturbation(0, i, j, k)) / dx[2] -
						     (perturbation(2, i + 1, j, k) - perturbation(2, i, j, k)) / dx[0];
		} else if (dir == quokka::direction::z) {
			// z-face: B_z = dA_y/dx - dA_x/dy
			const double z_face = prob_lo[2] + k * dx[2];
			const double x_hi = prob_lo[0] + (i + 1) * dx[0];
			const double x_lo = prob_lo[0] + i * dx[0];
			const double y_hi = prob_lo[1] + (j + 1) * dx[1];
			const double y_lo = prob_lo[1] + j * dx[1];
			const double vecpot_y_at_x_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * x_hi) + std::cos(seed_wavenumber * z_face));
			const double vecpot_y_at_x_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * x_lo) + std::cos(seed_wavenumber * z_face));
			const double vecpot_x_at_y_hi = seed_vecpot_amplitude * (std::sin(seed_wavenumber * z_face) + std::cos(seed_wavenumber * y_hi));
			const double vecpot_x_at_y_lo = seed_vecpot_amplitude * (std::sin(seed_wavenumber * z_face) + std::cos(seed_wavenumber * y_lo));
			state_fc(i, j, k, b_index) = (vecpot_y_at_x_hi - vecpot_y_at_x_lo) / dx[0] - (vecpot_x_at_y_hi - vecpot_x_at_y_lo) / dx[1] +
						     (perturbation(1, i + 1, j, k) - perturbation(1, i, j, k)) / dx[0] -
						     (perturbation(0, i, j + 1, k) - perturbation(0, i, j, k)) / dx[1];
		}
	});
}

auto problem_main() -> int
{
	amrex::ParmParse const pp("setup");
	pp.query("seed_b_wavenumber", seed_b_wavenumber);
	pp.query("seed_b_fraction", seed_b_fraction);
	pp.query("seed_b_perturbation_fraction", seed_b_perturbation_fraction);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
	    !pp.contains("helical_seed_peak_wavenumber"),
	    "setup.helical_seed_peak_wavenumber was replaced by setup.helical_seed_kpeak and the other power_randomphase_hel parameters");
	pp.query("helical_seed_kpeak", helical_seed_params.kpeak);
	pp.query("helical_seed_initpower", helical_seed_params.initpower);
	pp.query("helical_seed_initpower2", helical_seed_params.initpower2);
	pp.query("helical_seed_nfact", helical_seed_params.nfact);
	pp.query("helical_seed_cutoff", helical_seed_params.cutoff);
	pp.query("helical_seed_ncutoff", helical_seed_params.ncutoff);
	pp.query("helical_seed_relhel", helical_seed_params.relhel);
	pp.query("helical_seed_random_amplitude", helical_seed_params.random_amplitude);
	pp.query("helical_seed_planar_field", helical_seed_params.planar_field);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((helical_seed_params.planar_field == 0) || (helical_seed_params.relhel == 0.0),
					 "setup.helical_seed_planar_field = 1 builds a field with no helicity; set setup.helical_seed_relhel = 0");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helical_seed_params.kpeak >= 0.0, "setup.helical_seed_kpeak must be non-negative");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(helical_seed_params.nfact > 0.0, "setup.helical_seed_nfact must be positive");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(helical_seed_params.relhel) <= 1.0, "setup.helical_seed_relhel must lie in [-1, 1]");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((helical_seed_params.kpeak == 0.0) || (seed_b_perturbation_fraction == 0.0),
					 "setup.helical_seed_kpeak replaces the ABC seed and cannot be combined with setup.seed_b_perturbation_fraction");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(seed_b_perturbation_fraction >= 0.0, "setup.seed_b_perturbation_fraction must be non-negative");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE((seed_b_perturbation_fraction == 0.0) || (seed_b_wavenumber >= 2),
					 "setup.seed_b_perturbation_fraction > 0 needs setup.seed_b_wavenumber >= 2, so that modes exist below the seed");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(seed_b_wavenumber > 0, "setup.seed_b_wavenumber must be a positive integer");
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(seed_b_fraction > 0.0,
					 "setup.seed_b_fraction must be a positive value; the default of 0 runs with zero magnetic seed field (pure-hydro)");

	amrex::ParmParse const pp_turb("turbulence");
	pp_turb.query("target_vdisp", target_vdisp);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(target_vdisp > 0.0,
					 "turbulence.target_vdisp must be a positive value; the default of 0 leaves the velocity field to decay "
					 "(and thus the magnetic field, too)");

	amrex::ParmParse const pp_geom("geometry");
	amrex::Vector<amrex::Real> prob_lo_vec;
	amrex::Vector<amrex::Real> prob_hi_vec;
	pp_geom.getarr("prob_lo", prob_lo_vec);
	pp_geom.getarr("prob_hi", prob_hi_vec);
	const double box_length = prob_hi_vec[0] - prob_lo_vec[0];
	const bool box_is_cubic = (std::abs(box_length - (prob_hi_vec[1] - prob_lo_vec[1])) < 1e-10 * box_length) &&
				  (std::abs(box_length - (prob_hi_vec[2] - prob_lo_vec[2])) < 1e-10 * box_length);
	int turbulence_enabled = 0;
	pp_turb.query("enabled", turbulence_enabled);
	const bool box_may_be_noncubic = (helical_seed_params.kpeak > 0.0) && (turbulence_enabled == 0);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(box_is_cubic || box_may_be_noncubic,
					 "geometry.prob_hi - geometry.prob_lo must be equal in all three dimensions; the ABC seed field and "
					 "turbulence driving assume a cubic box (only setup.helical_seed_kpeak without driving supports a non-cubic box)");
	initializePerturbationModes(box_length);

	double turbulence_length = 0.0;
	pp_turb.query("length", turbulence_length);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(box_length - turbulence_length) < 1e-10 * turbulence_length,
					 "turbulence.length must match the box length (geometry.prob_hi[0] - geometry.prob_lo[0]); the seed "
					 "field and turbulence driving must agree on the box length");

	int ampl_auto_adjust = 0;
	pp_turb.query("ampl_auto_adjust", ampl_auto_adjust);
	if (ampl_auto_adjust == 1) {
		amrex::Print() << "WARNING: turbulence.ampl_auto_adjust = 1 is set. The driving amplitude will vary over time, "
				  "which makes it harder to interpret the dynamo growth rate. Consider setting ampl_auto_adjust = 0 "
				  "and tuning ampl_factor manually to achieve the desired velocity dispersion.\n";
	}

	auto BCs_cc = quokka::BC<MHDSmallScaleDynamo>(quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<MHDSmallScaleDynamo>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<MHDSmallScaleDynamo> sim(BCs_cc, BCs_fc);

	sim.setInitialConditions();
	helical_seed_vector_potential_mf.reset();

	sim.evolve();

	return 0;
}
