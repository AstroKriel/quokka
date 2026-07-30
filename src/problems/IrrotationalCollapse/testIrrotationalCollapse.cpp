//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testIrrotationalCollapse.cpp
/// \brief Self-gravitating collapse of an initially irrotational turbulent
///        velocity field in a periodic box, following Brandenburg, Ntormousi &
///        Schober 2026 (arXiv:2607.01207). Forked from the StarCluster problem.
///
/// The velocity field is set to a uniform-amplitude, curl-free turbulent field
/// (generated with perturbation.py --f_solenoidal=0.0). Unlike StarCluster, the
/// velocity is NOT weighted by the local density and NOT masked to the cloud:
/// multiplying an irrotational field by a spatially varying factor injects
/// vorticity, so the amplitude must be spatially uniform to keep the field
/// curl-free. The amplitude is set directly by u_ini (= Mach number for the
/// isothermal EOS with cs = 1), decoupled from the density profile.
///
#include "hydro/hydro_system.hpp"
#include "math/interpolate.hpp"
#include <fstream>
#include <limits>
#include <memory>
#include <random>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_BLassert.H"
#include "AMReX_Config.H"
#include "AMReX_FabArrayUtility.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_SPACE.H"
#include "AMReX_TableData.H"

#include "QuokkaSimulation.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "turbulence/TurbDataReader.hpp"
#include "util/BC.hpp"

using amrex::Real;

struct IrrotationalCollapse {
};

template <> struct quokka::EOS_Traits<IrrotationalCollapse> {
	static constexpr double gamma = 5. / 3.; // isentropic index (paper uses gamma = 5/3)
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct HydroSystem_Traits<IrrotationalCollapse> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct Physics_Traits<IrrotationalCollapse> : DefaultPhysicsTraits {
	// cell-centred
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_self_gravity_enabled = true;
	// face-centred
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr amrex::Real gravitational_constant = 1.0;
};

template <> struct SimulationData<IrrotationalCollapse> {
	// real-space irrotational velocity perturbation field
	amrex::TableData<Real, 3> dvx;
	amrex::TableData<Real, 3> dvy;
	amrex::TableData<Real, 3> dvz;
	Real dv_rms_generated{};
	Real rescale_factor{};

	// cloud parameters
	Real R_sphere{};
	Real rho_sphere{};
	Real u_ini{};	    // target rms velocity amplitude
	Real sound_speed{}; // reference sound speed at rho_sphere (Mach = u_ini / sound_speed)
};

template <> void QuokkaSimulation<IrrotationalCollapse>::preCalculateInitialConditions()
{
	static bool isSamplingDone = false;
	if (!isSamplingDone) {
		// read perturbations from file
		turb_data turbData;
		amrex::ParmParse const pp("perturb");
		std::string turbdata_filename;
		pp.query("filename", turbdata_filename);
		initialize_turbdata(turbData, turbdata_filename);

		// copy to pinned memory
		auto pinned_dvx = get_tabledata(turbData.dvx);
		auto pinned_dvy = get_tabledata(turbData.dvy);
		auto pinned_dvz = get_tabledata(turbData.dvz);

		// compute the rms of the generated field so we can normalise it to
		// exactly u_ini (the generator already targets unit rms, but normalise
		// explicitly so the amplitude is set by u_ini alone)
		userData_.dv_rms_generated = computeRms(pinned_dvx, pinned_dvy, pinned_dvz);
		amrex::Print() << "rms dv = " << userData_.dv_rms_generated << "\n";

		const Real rms_dv_target = userData_.u_ini;
		const Real rms_Mach_target = rms_dv_target / userData_.sound_speed;
		userData_.rescale_factor = rms_dv_target / userData_.dv_rms_generated;
		amrex::Print() << "rms Mach target = " << rms_Mach_target << "\n";

		// copy to GPU
		userData_.dvx.resize(pinned_dvx.lo(), pinned_dvx.hi());
		userData_.dvx.copy(pinned_dvx);

		userData_.dvy.resize(pinned_dvy.lo(), pinned_dvy.hi());
		userData_.dvy.copy(pinned_dvy);

		userData_.dvz.resize(pinned_dvz.lo(), pinned_dvz.hi());
		userData_.dvz.copy(pinned_dvz);

		isSamplingDone = true;
	}
}

template <> void QuokkaSimulation<IrrotationalCollapse>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// set initial conditions
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx = grid_elem.dx_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi = grid_elem.prob_hi_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	amrex::Real const x0 = prob_lo[0] + 0.5 * (prob_hi[0] - prob_lo[0]);
	amrex::Real const y0 = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
	amrex::Real const z0 = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);

	// cloud parameters (density profile only; the velocity is independent)
	const double rho_min = 0.01 * userData_.rho_sphere;
	const double rho_max = userData_.rho_sphere;
	const double R_sphere = userData_.R_sphere;
	const double R_smooth = 0.05 * R_sphere;
	const double renorm_amp = userData_.rescale_factor;

	// isentropic pressure normalisation: p = p_ref (rho/rho_ref)^gamma, with the
	// reference sound speed cs_ref set at rho_ref (so cs = cs_ref where rho = rho_ref)
	const double gamma = quokka::EOS_Traits<IrrotationalCollapse>::gamma;
	const double rho_ref = userData_.rho_sphere;
	const double cs_ref = userData_.sound_speed;
	const double p_ref = rho_ref * cs_ref * cs_ref / gamma;

	auto const &dvx = userData_.dvx.const_table();
	auto const &dvy = userData_.dvy.const_table();
	auto const &dvz = userData_.dvz.const_table();

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		amrex::Real const x = prob_lo[0] + (i + static_cast<amrex::Real>(0.5)) * dx[0];
		amrex::Real const y = prob_lo[1] + (j + static_cast<amrex::Real>(0.5)) * dx[1];
		amrex::Real const z = prob_lo[2] + (k + static_cast<amrex::Real>(0.5)) * dx[2];
		amrex::Real const r = std::sqrt(std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2));

		// density: smoothed top-hat sphere on a periodic background
		double const rho = std::max(rho_min, rho_max * ((std::tanh((R_sphere - r) / R_smooth) + 1.0) / 2.0));
		AMREX_ASSERT(!std::isnan(rho));

		// velocity: uniform-amplitude irrotational field. Scaling by the
		// spatially constant renorm_amp preserves curl(v) = 0; a density- or
		// radius-dependent factor would not.
		//
		// The turbulence reader wraps the C-order HDF5 buffer as a
		// Fortran-order Table3D without transposing (see the WARNING in
		// TurbDataReader.cpp), so each component arrives with its spatial axes
		// reversed: dv*(i,j,k) = pert*[k,j,i]. For a scalar driving field this
		// is harmless, but for a vector field it breaks curl(v) = 0 unless the
		// components are swapped to match the axis reversal. Assigning x <- dvz
		// and z <- dvx is the x<->z relabel that restores the irrotational
		// field (verified: solenoidal KE fraction drops from 0.52 to 4e-7).
		double const vx = renorm_amp * dvz(i, j, k);
		double const vy = renorm_amp * dvy(i, j, k);
		double const vz = renorm_amp * dvx(i, j, k);

		// isentropic pressure on the initial adiabat, and the corresponding
		// internal energy; the total gas energy also carries the kinetic part
		double const P = p_ref * std::pow(rho / rho_ref, gamma);
		double const eint = quokka::EOS<IrrotationalCollapse>::ComputeEintFromPres(rho, P);
		double const ekin = 0.5 * rho * (vx * vx + vy * vy + vz * vz);

		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::x1Momentum_index) = rho * vx;
		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::x2Momentum_index) = rho * vy;
		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::x3Momentum_index) = rho * vz;
		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::energy_index) = eint + ekin;
		state_cc(i, j, k, HydroSystem<IrrotationalCollapse>::internalEnergy_index) = eint;
	});
}

template <> void QuokkaSimulation<IrrotationalCollapse>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// refine on Jeans length
	const int N_cells = 4;			      // inverse of the 'Jeans number' [Truelove et al. (1997)]
	const amrex::Real cs = userData_.sound_speed; // reference sound speed for the Jeans criterion
	const amrex::Real dx = geom[lev].CellSizeArray()[0];
	const amrex::Real G = Gconst_;

	auto const &state = state_new_cc_[lev].const_arrays();
	auto tag = tags.arrays();

	amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		Real const rho = state[bx](i, j, k, HydroSystem<IrrotationalCollapse>::density_index);
		const amrex::Real l_Jeans = cs * std::sqrt(M_PI / (G * rho));

		if (l_Jeans < (N_cells * dx)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
}

template <>
void QuokkaSimulation<IrrotationalCollapse>::ComputeDerivedVar(int /*lev*/, std::string const &dname, amrex::MultiFab &mf, const int ncomp_cc_in,
							       amrex::MultiFab const &state_cc,
							       amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> const & /*state_fc*/) const
{
	// compute derived variables and save in 'mf'
	if (dname == "log_density") {
		const int ncomp = ncomp_cc_in;
		auto const &state = state_cc.const_arrays();
		auto output = mf.arrays();

		amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
			Real const rho = state[bx](i, j, k, HydroSystem<IrrotationalCollapse>::density_index);
			output[bx](i, j, k, ncomp) = std::log10(rho);
		});
	}
}

auto problem_main() -> int
{
	// periodic boundary conditions (the FFT velocity field is periodic, and
	// the paper uses a periodic box with self-gravity + mean subtraction)
	auto BCs_cc = quokka::BC<IrrotationalCollapse>(quokka::BCType::int_dir,	 // x: periodic
						       quokka::BCType::int_dir,	 // y: periodic
						       quokka::BCType::int_dir); // z: periodic

	// read problem parameters
	amrex::ParmParse const pp("perturb");

	Real R_sphere{};
	pp.query("cloud_radius", R_sphere);

	Real rho_sphere{};
	pp.query("cloud_density", rho_sphere);

	Real u_ini{};
	pp.query("velocity_amplitude", u_ini);

	Real sound_speed = 1.0;
	pp.query("sound_speed", sound_speed);

	// Problem initialization
	QuokkaSimulation<IrrotationalCollapse> sim(BCs_cc);
	sim.densityFloor_ = 0.01;

	sim.userData_.R_sphere = R_sphere;
	sim.userData_.rho_sphere = rho_sphere;
	sim.userData_.u_ini = u_ini;
	sim.userData_.sound_speed = sound_speed;

	// initialize
	sim.setInitialConditions();

	// evolve
	sim.evolve();

	int const status = 0;
	return status;
}
