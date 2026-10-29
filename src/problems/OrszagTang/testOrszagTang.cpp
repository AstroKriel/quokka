//==============================================================================
// Copyright 2025 Ben Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testOrszagTang.cpp
/// \brief
///   This problem is based on the implementation here:
///   https://github.com/PrincetonUniversity/athena/blob/master/src/pgen/orszag_tang.cpp.
///	  (Phil Hopkins made several typos on this page, do not use:
///	  https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html)
///

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <string>

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_Reduce.H"
#include "AMReX_TagBox.H"

#include "QuokkaSimulation.hpp"
#include "grid.hpp"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

struct OrszagTang {};

AMREX_ENUM(RefineOn, CurrentDensity, PressureGradient); // NOLINT

template <> struct quokka::EOS_Traits<OrszagTang> {
	static constexpr double gamma = 5. / 3.;
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<OrszagTang> : DefaultPhysicsTraits {
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr bool is_hydro_enabled = true;
	static constexpr bool is_mhd_enabled = true;
};

constexpr double B0 = 1.0 / gcem::sqrt(4.0 * PI);

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto A_z(double x, double y) -> double
{
	return B0 / (4.0 * M_PI) * (std::cos(4.0 * M_PI * x) - 2.0 * std::cos(2.0 * M_PI * y));
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_x(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return (A_z(xL, yL + dx[1]) - A_z(xL, yL)) / dx[1];
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto B_y(double xL, double yL, const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx) -> double
{
	return -(A_z(xL + dx[0], yL) - A_z(xL, yL)) / dx[0];
};

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	// extract grid information
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;

	constexpr double gamma_gas = quokka::EOS_Traits<OrszagTang>::gamma;
	constexpr double rho0 = 25. / (36. * M_PI);
	constexpr double P0 = 5. / (12. * M_PI);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double x = prob_lo[0] + ((i + 0.5) * dx[0]);
		const double y = prob_lo[1] + ((j + 0.5) * dx[1]);

		const double vx = std::sin(2 * M_PI * y);
		const double vy = -std::sin(2 * M_PI * x);

		const double Bx = 0.5 * (B_x(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_x(x + 0.5 * dx[0], y - 0.5 * dx[1], dx));
		const double By = 0.5 * (B_y(x - 0.5 * dx[0], y - 0.5 * dx[1], dx) + B_y(x - 0.5 * dx[0], y + 0.5 * dx[1], dx));

		const double Ekin = 0.5 * rho0 * (vx * vx + vy * vy);
		const double Eint = P0 / (gamma_gas - 1.0);
		const double Emag = 0.5 * (Bx * Bx + By * By);

		state_cc(i, j, k, HydroSystem<OrszagTang>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x1Momentum_index) = rho0 * vx;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x2Momentum_index) = rho0 * vy;
		state_cc(i, j, k, HydroSystem<OrszagTang>::x3Momentum_index) = 0;
		state_cc(i, j, k, HydroSystem<OrszagTang>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<OrszagTang>::energy_index) = Eint + Ekin + Emag;
	});
}

template <> void QuokkaSimulation<OrszagTang>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const double xL = prob_lo[0] + (i * dx[0]);
		const double yL = prob_lo[1] + (j * dx[1]);

		if (dir == quokka::direction::x) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_x(xL, yL, dx);
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = B_y(xL, yL, dx);
		} else if (dir == quokka::direction::z) {
			state_fc(i, j, k, Physics_Indices<OrszagTang>::mhdFirstIndex) = 0;
		}
	});
}

// average the two faces of cell (i, j, k) normal to world direction w
AMREX_GPU_DEVICE AMREX_FORCE_INLINE auto cellCentredB(std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const &state_fc, int w, int i, int j, int k)
    -> amrex::Real
{
	constexpr int b_comp = Physics_Indices<OrszagTang>::mhdFirstIndex;
	const amrex::IntVect lower_face{AMREX_D_DECL(i, j, k)};
	const amrex::IntVect upper_face = lower_face + amrex::IntVect::TheDimensionVector(w);
	return 0.5 * (state_fc[w](lower_face, b_comp) + state_fc[w](upper_face, b_comp));
}

template <> void QuokkaSimulation<OrszagTang>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	amrex::ParmParse const pp("setup");
	RefineOn refine_based_on{};
	pp.get("refine_based_on", refine_based_on);
	amrex::Real refine_threshold = 0.0;
	pp.get("refine_threshold", refine_threshold);
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(refine_threshold > 0.0, "setup.refine_threshold must be > 0.");
	// ghost cells still hold data from before the last advance; refill them before reading neighbours
	fillBoundaryConditions(state_new_cc_[lev], state_new_cc_[lev], lev, tNew_[lev], quokka::centering::cc, quokka::direction::na, InterpHookNone,
			       InterpHookNone, FillPatchType::fillpatch_function);
	for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
		fillBoundaryConditions(state_new_fc_[lev][idim], state_new_fc_[lev][idim], lev, tNew_[lev], quokka::centering::fc,
				       static_cast<quokka::direction>(idim), InterpHookNone, InterpHookNone, FillPatchType::fillpatch_function);
	}
	const auto dx = geom[lev].CellSizeArray();
	const amrex::Real dx_min = std::min({dx[0], dx[1], dx[2]});
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		const auto tag = tags.array(mfi);
		const auto state_cc = state_new_cc_[lev].const_array(mfi);
		const std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> state_fc{
		    state_new_fc_[lev][0].const_array(mfi), state_new_fc_[lev][1].const_array(mfi), state_new_fc_[lev][2].const_array(mfi)};
		if (refine_based_on == RefineOn::CurrentDensity) {
			// |curl B| times the smallest cell width, relative to the initial field strength B0
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real dbz_dy = (cellCentredB(state_fc, 2, i, j + 1, k) - cellCentredB(state_fc, 2, i, j - 1, k)) / (2.0 * dx[1]);
				const amrex::Real dby_dz = (cellCentredB(state_fc, 1, i, j, k + 1) - cellCentredB(state_fc, 1, i, j, k - 1)) / (2.0 * dx[2]);
				const amrex::Real dbx_dz = (cellCentredB(state_fc, 0, i, j, k + 1) - cellCentredB(state_fc, 0, i, j, k - 1)) / (2.0 * dx[2]);
				const amrex::Real dbz_dx = (cellCentredB(state_fc, 2, i + 1, j, k) - cellCentredB(state_fc, 2, i - 1, j, k)) / (2.0 * dx[0]);
				const amrex::Real dby_dx = (cellCentredB(state_fc, 1, i + 1, j, k) - cellCentredB(state_fc, 1, i - 1, j, k)) / (2.0 * dx[0]);
				const amrex::Real dbx_dy = (cellCentredB(state_fc, 0, i, j + 1, k) - cellCentredB(state_fc, 0, i, j - 1, k)) / (2.0 * dx[1]);
				const amrex::Real current_x = dbz_dy - dby_dz;
				const amrex::Real current_y = dbx_dz - dbz_dx;
				const amrex::Real current_z = dby_dx - dbx_dy;
				const amrex::Real current_magnitude = std::sqrt(current_x * current_x + current_y * current_y + current_z * current_z);
				if (current_magnitude * dx_min / B0 > refine_threshold) {
					tag(i, j, k) = amrex::TagBox::SET;
				}
			});
		} else if (refine_based_on == RefineOn::PressureGradient) {
			// gas-pressure jump across one cell, relative to the local pressure
			amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real pressure = HydroSystem<OrszagTang>::ComputePressure(state_cc, i, j, k, &state_fc);
				const amrex::Real dp_x = 0.5 * (HydroSystem<OrszagTang>::ComputePressure(state_cc, i + 1, j, k, &state_fc) -
								HydroSystem<OrszagTang>::ComputePressure(state_cc, i - 1, j, k, &state_fc));
				const amrex::Real dp_y = 0.5 * (HydroSystem<OrszagTang>::ComputePressure(state_cc, i, j + 1, k, &state_fc) -
								HydroSystem<OrszagTang>::ComputePressure(state_cc, i, j - 1, k, &state_fc));
				const amrex::Real dp_z = 0.5 * (HydroSystem<OrszagTang>::ComputePressure(state_cc, i, j, k + 1, &state_fc) -
								HydroSystem<OrszagTang>::ComputePressure(state_cc, i, j, k - 1, &state_fc));
				if (std::sqrt(dp_x * dp_x + dp_y * dp_y + dp_z * dp_z) / pressure > refine_threshold) {
					tag(i, j, k) = amrex::TagBox::SET;
				}
			});
		}
	}
	int tag_full_columns = 0;
	pp.query("tag_full_columns", tag_full_columns);
	if (tag_full_columns == 0) {
		return;
	}
	// a column is tagged everywhere if any of its cells is tagged, so refined patches span the full z-extent
	const amrex::Box &domain = geom[lev].Domain();
	const int nx = domain.length(0);
	const int x_lo = domain.smallEnd(0);
	const int y_lo = domain.smallEnd(1);
	const int num_columns = nx * domain.length(1);
	amrex::Gpu::DeviceVector<int> column_is_tagged(num_columns, 0);
	int *column_is_tagged_ptr = column_is_tagged.data();
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const auto tag = tags.const_array(mfi);
		amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (tag(i, j, k) == amrex::TagBox::SET) {
				column_is_tagged_ptr[((j - y_lo) * nx) + (i - x_lo)] = 1;
			}
		});
	}
	amrex::Vector<int> column_is_tagged_host(num_columns);
	amrex::Gpu::copy(amrex::Gpu::deviceToHost, column_is_tagged.begin(), column_is_tagged.end(), column_is_tagged_host.begin());
	amrex::ParallelDescriptor::ReduceIntMax(column_is_tagged_host.data(), num_columns);
	amrex::Gpu::copy(amrex::Gpu::hostToDevice, column_is_tagged_host.begin(), column_is_tagged_host.end(), column_is_tagged.begin());
	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const auto tag = tags.array(mfi);
		amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (column_is_tagged_ptr[((j - y_lo) * nx) + (i - x_lo)] == 1) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

// the solution is z-invariant, so any z-momentum, z-field or z-variation of density measures broken symmetry
template <> void QuokkaSimulation<OrszagTang>::computeAfterTimestep()
{
	constexpr int density_comp = HydroSystem<OrszagTang>::density_index;
	for (int lev = 0; lev <= finest_level; ++lev) {
		const amrex::MultiFab &state_cc = state_new_cc_[lev];
		const amrex::Real max_abs_momentum_z = state_cc.norminf(HydroSystem<OrszagTang>::x3Momentum_index);
		const amrex::Real max_abs_field_z = state_new_fc_[lev][2].norminf(Physics_Indices<OrszagTang>::mhdFirstIndex);
		amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
		amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
		for (amrex::MFIter mfi(state_cc); mfi.isValid(); ++mfi) {
			const amrex::Box &box = mfi.validbox();
			const int k_top = box.bigEnd(2);
			const auto state = state_cc.const_array(mfi);
			reduce_op.eval(box, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real> {
				if (k == k_top) {
					return {0.0};
				}
				return {std::abs(state(i, j, k + 1, density_comp) - state(i, j, k, density_comp)) / state(i, j, k, density_comp)};
			});
		}
		amrex::Real max_density_z_jump = amrex::get<0>(reduce_data.value());
		amrex::ParallelDescriptor::ReduceRealMax(max_density_z_jump);
		amrex::Print() << "[z-symmetry] step " << istep[0] << " t = " << std::setprecision(6) << tNew_[0] << " level " << lev << std::scientific
			       << std::setprecision(3) << " density_z_jump " << max_density_z_jump << " max_abs_momentum_z " << max_abs_momentum_z
			       << " max_abs_field_z " << max_abs_field_z << std::defaultfloat << "\n";
	}
}

namespace
{
enum class Rot180Parity { Even, Odd };

// map a single-axis index under 180-degree rotation about the domain centre; generalised for both
// cell-centred (range [0, n-1] -> n-1-i) and face/nodal (range [0, n] -> n-i) axes.
AMREX_FORCE_INLINE auto rot180Index(int i, int n, bool is_nodal) -> int { return is_nodal ? (n - i) : (n - 1 - i); }

// gather `comp` of `mf` onto a single box on the IO processor (mf may be domain-decomposed across
// any number of boxes/ranks) and check that every cell's value exactly matches (Even) or exactly
// negates (Odd) its 180-degree-rotation partner.
auto checkRot180Symmetry(const amrex::MultiFab &mf, int comp, const amrex::Box &domain, Rot180Parity parity, const std::string &label) -> int
{
	const amrex::BoxArray ba_single(amrex::convert(domain, mf.ixType()));
	const amrex::DistributionMapping dm_single(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
	amrex::MultiFab mf_single(ba_single, dm_single, 1, 0);
	mf_single.ParallelCopy(mf, comp, 0, 1);

	int num_diffs = 0;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		const amrex::MFIter mfi(mf_single);
		const amrex::Box &box = mfi.validbox();
#ifdef AMREX_USE_GPU
		amrex::FArrayBox host_fab(box, 1, amrex::The_Pinned_Arena());
		static_cast<void>(mf_single[mfi].template copyToMem<amrex::RunOn::Device>(box, 0, 1, host_fab.dataPtr()));
		amrex::Gpu::synchronize();
#else
		amrex::FArrayBox host_fab(box, 1);
		static_cast<void>(mf_single[mfi].template copyToMem<amrex::RunOn::Host>(box, 0, 1, host_fab.dataPtr()));
#endif
		const auto &field = host_fab.const_array();
		const amrex::IntVect is_nodal = mf.ixType().toIntVect();
		const std::array<int, 2> n{domain.length(0), domain.length(1)};

		for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
			for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
				for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
					const int i_rot = rot180Index(i, n[0], is_nodal[0] != 0);
					const int j_rot = rot180Index(j, n[1], is_nodal[1] != 0);
					if ((i_rot < i) || (i_rot == i && j_rot < j)) {
						continue; // each pair only needs checking once
					}
					const amrex::Real val = field(i, j, k);
					const amrex::Real val_rot = field(i_rot, j_rot, k);
					const amrex::Real expected = (parity == Rot180Parity::Even) ? val : -val;
					if (val_rot != expected) {
						++num_diffs;
						amrex::Print()
						    << "[rot180] " << label << " mismatch: (" << i << "," << j << "," << k << ")=" << std::setprecision(17)
						    << val << " vs (" << i_rot << "," << j_rot << "," << k << ")=" << std::setprecision(17) << val_rot
						    << " (expected " << std::setprecision(17) << expected << ")\n";
					}
				}
			}
		}

		if (num_diffs == 0) {
			amrex::Print() << "[rot180] " << label << ": exact match under 180-degree rotation.\n";
		} else {
			amrex::Print() << "[rot180] " << label << ": found " << num_diffs << " mismatched cell pairs under 180-degree rotation.\n";
		}
	}
	return num_diffs;
}
} // namespace

template <> void QuokkaSimulation<OrszagTang>::computeAfterEvolve(amrex::Vector<amrex::Real> & /*initSumCons*/)
{
	if (max_level > 0) {
		amrex::Print() << "[rot180] skipped: AMR (amr.max_level > 0) does not preserve bit-exact point-reflection symmetry.\n";
		return;
	}
	// the Orszag-Tang vortex's IC is point-symmetric about the domain centre, so the exact solution stays
	// bit-exactly rot180-symmetric for all time; a floating-point non-associativity bug in
	// reconstruction, EMF averaging, or the Riemann solver breaks this (see PR #1997).
	const amrex::Box &domain = Geom(0).Domain();
	int num_diffs = 0;
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::density_index, domain, Rot180Parity::Even, "density");
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::x1Momentum_index, domain, Rot180Parity::Odd, "x1Momentum");
	num_diffs += checkRot180Symmetry(state_new_cc_[0], HydroSystem<OrszagTang>::x2Momentum_index, domain, Rot180Parity::Odd, "x2Momentum");
	num_diffs += checkRot180Symmetry(state_new_fc_[0][0], Physics_Indices<OrszagTang>::mhdFirstIndex, domain, Rot180Parity::Odd, "Bx");
	num_diffs += checkRot180Symmetry(state_new_fc_[0][1], Physics_Indices<OrszagTang>::mhdFirstIndex, domain, Rot180Parity::Odd, "By");

	if (amrex::ParallelDescriptor::IOProcessor()) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_diffs == 0, "OrszagTang rot180 symmetry check failed: the MHD solver broke bit-exact point-reflection "
								 "symmetry (see PR #1997); this indicates a regression in floating-point term-pairing for "
								 "WENO reconstruction, EMF averaging, or HLLD.");
	}
}

auto problem_main() -> int
{
	QuokkaSimulation<OrszagTang> sim;
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
