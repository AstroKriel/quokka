#ifndef MHD_SYSTEM_HPP_ // NOLINT
#define MHD_SYSTEM_HPP_
//==============================================================================
// ...
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file mhd_system.hpp
/// \brief Defines a class for solving the MHD equations.
///

// c++ headers

// library headers

// internal headers
#include "AMReX_ParmParse.H"
#include "AMReX_Print.H"
#include "hydro_system.hpp"
#include "hyperbolic_system.hpp"
#include "physics_info.hpp"
#include "physics_numVars.hpp"

/// Class for a MHD system of conservation laws
template <typename problem_t> class MHDSystem : public HyperbolicSystem<problem_t>
{
      public:
	static constexpr int nvar_per_dim_ = Physics_NumVars::numMHDVars_per_dim;
	static constexpr int nvar_tot_ = Physics_NumVars::numMHDVars_tot;

	enum varIndex_perDim {
		bfield_index = Physics_Indices<problem_t>::mhdFirstIndex,
	};

	static void ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
			       std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds,
			       int nghost_fc);

	static void ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &reconstructRange,
				  int reconstructionOrder);

  static void SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, const double dt);
};

template <typename problem_t>
void MHDSystem<problem_t>::ComputeEMF(std::array<amrex::MultiFab, AMREX_SPACEDIM> &ec_mf_emf_comps, amrex::MultiFab const &cc_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_cVars,
				      std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fcx_mf_fspds, const int nghost_fc)
{
	const int reconstructionOrder = 3;
  const int nghost_cc = 4; // we only need 4 cc ghost cells when reconstructing cc->fc->ec using PPM

  // loop over each box-array on the level
	// note: all the different centerings still have the same distribution mapping, so it is fine for us to attach our looping to cc FArrayBox
  // note: cell-centered (cc), face-centered (fc), and edge-centered (ec) data all have a different number of cells
  for (amrex::MFIter mfi(cc_mf_cVars); mfi.isValid(); ++mfi) {
    const amrex::Box &box_cc = mfi.validbox();
    const amrex::Box &box_r = amrex::grow(box_cc, 1); // reconstruction range

		// In this function we distinguish between world (w:3), array (i:2), quandrant (q:4), and component (x:3) index-ing by using prefixes. We will use
		// the prefix x- when the w- and i- indexes are the same. We also choose to minimise the storage footprint by only computing and holding onto the
		// quantities required for calculating the EMF in the w-direction. This inadvertently leads to duplicate computation, but allows us to significantly
		// reduces the total memory used, which is a much bigger bottleneck.

		// extract cell-centered velocity fields
		// indexing: field[3: x-component]
		std::array<amrex::FArrayBox, 3> cc_fabs_Ux;
		const amrex::Box &box_cc_U = amrex::grow(box_cc, nghost_cc);
		cc_fabs_Ux[0].resize(box_cc_U, 1);
		cc_fabs_Ux[1].resize(box_cc_U, 1);
		cc_fabs_Ux[2].resize(box_cc_U, 1);
		const auto &cc_a4_Ux0 = cc_fabs_Ux[0].array();
		const auto &cc_a4_Ux1 = cc_fabs_Ux[1].array();
		const auto &cc_a4_Ux2 = cc_fabs_Ux[2].array();
		const auto &cc_a4_cVars = cc_mf_cVars[mfi].const_array();
		amrex::ParallelFor(box_cc_U, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
			const auto rho = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::density_index);
			const auto px1 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
			const auto px2 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
			const auto px3 = cc_a4_cVars(i, j, k, HydroSystem<problem_t>::x3Momentum_index);
			cc_a4_Ux0(i, j, k) = px1 / rho;
			cc_a4_Ux1(i, j, k) = px2 / rho;
			cc_a4_Ux2(i, j, k) = px3 / rho;
		});

		// indexing: field[3: x-component/x-face]
		// create a view of all the b-field data (+ghost cells; do not make another copy)
		std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
		    amrex::FArrayBox(fcx_mf_cVars[0][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[1][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		    amrex::FArrayBox(fcx_mf_cVars[2][mfi], amrex::make_alias, MHDSystem<problem_t>::bfield_index, 1),
		};

		// compute the magnetic flux through each cell-face
		for (int wsolve = 0; wsolve < 3; ++wsolve) {
			// to solve for the magnetic flux through each face we compute the line integral of the EMF around the cell-face.
			// so let's solve for the EMF along each of the two edges along the edge of the cell-face
			int iedge_rel2face = 0;
      // for each of the two cell-edges on the cell-face
      // we are doing redundant compute. only need to look at one edge for each face: there is a one-to-one mapping.

      // define the two directions we need to extrapolate cell-centered velocity fields to get them to the cell-edge
      // we will want to compute E2 = (U0 * B1 - U1 * B0) along the cell-edge
      std::array<int, 2> w_extrap_dirs = {
          wsolve, // dir-0: w-direction of the cell-face being solved for (relative to the cell-center)
          (wsolve + iedge_rel2face + 1) % 3 // dir-1: w-direction of the cell-edge being solved for (relative to the cell-face)
      };
      std::array<amrex::IntVect, 2> ivecs_cc2ec = {
        amrex::IntVect::TheDimensionVector(w_extrap_dirs[0]),
        amrex::IntVect::TheDimensionVector(w_extrap_dirs[1])
      };
      const amrex::IntVect ivec_cc2ec = ivecs_cc2ec[0] + ivecs_cc2ec[1];
      const amrex::Box box_ec = amrex::convert(box_cc, ivec_cc2ec);
      const amrex::Box box_ec_r = amrex::grow(box_ec, 1);

      // initialise FArrayBox for storing the temporary edge-centered velocity fields created in each permutation of reconstructing from the cell-face
      // indexing: field[2: i-side of edge]
      std::array<amrex::FArrayBox, 2> ec_fabs_U_ieside;
      // define the four possible velocity field quantities that could be reconstructed at the cell-edge
      // also define the temporary velocity field quantities that will be used for computing the extrapolation
      ec_fabs_U_ieside[0].resize(box_ec_r, 1);
      ec_fabs_U_ieside[1].resize(box_ec_r, 1);
      // indexing: field[2: i-compnent][2: i-side of edge]
      // note: magnetic field components cannot be discontinuous along themselves (i.e., either side of the face where they are
      // stored), so there are only two possible values (sides), rather than four (quadrants of) possible reconstructed values
      std::array<std::array<amrex::FArrayBox, 2>, 2> ec_fabs_Bi_ieside;
      // initialise FArrayBox for storing the edge-centered velocity fields averaged across the two extrapolation permutations
      // indexing: field[2: i-compnent][4: quadrant around edge]
      std::array<std::array<amrex::FArrayBox, 4>, 2> ec_fabs_Ui_q;
      // define quantities
      for (int icomp = 0; icomp < 2; ++icomp) {
        ec_fabs_Bi_ieside[icomp][0].resize(box_ec_r, 1);
        ec_fabs_Bi_ieside[icomp][1].resize(box_ec_r, 1);
        for (int iquad = 0; iquad < 4; ++iquad) {
          ec_fabs_Ui_q[icomp][iquad].resize(box_ec_r, 1);
          ec_fabs_Ui_q[icomp][iquad].setVal(0.0);
        }
      }

      // extrapolate the two required cell-centered velocity field components to the cell-edge
      // there are two possible permutations for doing this: getting cell-centered quanties to a cell-edge
      // first is cc->fc[dir-0]->ec and second is cc->fc[dir-1]->ec
      for (int iperm = 0; iperm < 2; ++iperm) {
        // for each permutation of extrapolating cc->fc->ec

        // define quantities
        const int w_extrap_dir2face = w_extrap_dirs[iperm];
        const int w_extrap_dir2edge = w_extrap_dirs[(iperm + 1) % 2];
        const auto dir2face = static_cast<FluxDir>(w_extrap_dir2face);
        const auto dir2edge = static_cast<FluxDir>(w_extrap_dir2edge);
        const amrex::IntVect ivec_cc2fc = amrex::IntVect::TheDimensionVector(w_extrap_dir2face);
        const amrex::IntVect ivec_fc2ec = amrex::IntVect::TheDimensionVector(w_extrap_dir2edge);
        const amrex::Box box_fc = amrex::convert(box_cc, ivec_cc2fc);
        // in anticipation for fc->ec reconstruction, we should expand the domain to include ghost cells in that fc->ec dimension
        const amrex::Box box_cc_U = amrex::grow(box_cc, (nghost_cc-1)*ivec_fc2ec); // the bounds will be grown by 1 in the reconstruct func.
        const amrex::Box box_fc_U = amrex::grow(box_fc, 1 + (nghost_cc-1)*ivec_fc2ec);

        // create temporary FArrayBox for storing the face-centered velocity fields reconstructed from the cell-center
        // indexing: field[2: i-compnent][2: i-side of face]
        std::array<std::array<amrex::FArrayBox, 2>, 2> fc_fabs_Ui_ifside;
        // extrapolate both required cell-centered velocity fields to the cell-edge
        for (int icomp = 0; icomp < 2; ++icomp) {
          const int wcomp = w_extrap_dirs[icomp];
          fc_fabs_Ui_ifside[icomp][0].resize(box_fc_U, 1);
          fc_fabs_Ui_ifside[icomp][1].resize(box_fc_U, 1);
          // extrapolate cell-centered velocity components to the cell-face
          MHDSystem<problem_t>::ReconstructTo(dir2face, cc_fabs_Ux[wcomp].array(), fc_fabs_Ui_ifside[icomp][0].array(), fc_fabs_Ui_ifside[icomp][1].array(), box_cc_U, reconstructionOrder);
          // extrapolate face-centered velocity components to the cell-edge
          for (int iface = 0; iface < 2; ++iface) {
            // reset values in temporary FArrayBox
            ec_fabs_U_ieside[0].setVal(0.0);
            ec_fabs_U_ieside[1].setVal(0.0);
            // extrapolate face-centered velocity component to the cell-edge
            MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Ui_ifside[icomp][iface].array(), ec_fabs_U_ieside[0].array(), ec_fabs_U_ieside[1].array(), box_fc, reconstructionOrder);
            // figure out which quadrant of the cell-edge this extrapolated velocity component corresponds with
            int iquad0 = -1;
            int iquad1 = -1;
            // note: quadrants are defined based on where the quantity sits relative to the edge (dir-0, dir-1):
            // (-,+) | (+,+)
            //   1   |   2
            // ------+------
            //   0   |   3
            // (-,-) | (+,-)
            if (iperm == 0) {
              iquad0 = (iface == 0) ? 0 : 3;
              iquad1 = (iface == 0) ? 1 : 2;
            } else {
              iquad0 = (iface == 0) ? 0 : 1;
              iquad1 = (iface == 0) ? 3 : 2;
            }
            ec_fabs_Ui_q[icomp][iquad0].plus(ec_fabs_U_ieside[0], 0, 0, 1);
            ec_fabs_Ui_q[icomp][iquad1].plus(ec_fabs_U_ieside[1], 0, 0, 1);
          }
        }
      }
      // finish averaging the two different ways for extrapolating velocity fields: cc->fc->ec
      for (int icomp = 0; icomp < 2; ++icomp) {
        for (int iquad = 0; iquad < 4; ++iquad) {
          ec_fabs_Ui_q[icomp][iquad].mult(0.5, 0, 1);
        }
      }

      // extrapolate the two required face-centered magnetic field components to the cell-edge
      for (int icomp = 0; icomp < 2; ++icomp) {
        const int w_extrap_dir2edge = w_extrap_dirs[(icomp + 1) % 2];
        const auto dir2edge = static_cast<FluxDir>(w_extrap_dir2edge);
        const int wcomp = w_extrap_dirs[icomp];
        const amrex::IntVect ivec_cc2fc = amrex::IntVect::TheDimensionVector(wcomp);
        const amrex::Box box_fc = amrex::convert(box_cc, ivec_cc2fc);
        // extrapolate face-centered magnetic components to the cell-edge
        MHDSystem<problem_t>::ReconstructTo(dir2edge, fc_fabs_Bx[wcomp].array(), ec_fabs_Bi_ieside[icomp][0].array(), ec_fabs_Bi_ieside[icomp][1].array(), box_fc, reconstructionOrder);

        int tmp = 0; // TODO: for debuging. remove
      }

      // indexing: field[4: quadrant around edge]
      std::array<amrex::FArrayBox, 4> ec_fabs_E_q;
      // compute the EMF along the cell-edge
      for (int iquad = 0; iquad < 4; ++iquad) {
        // define EMF FArrayBox
        ec_fabs_E_q[iquad].resize(box_ec, 1);
        // extract relevant velocity and magnetic field components
        const auto &U0_qi = ec_fabs_Ui_q[0][iquad].const_array();
        const auto &U1_qi = ec_fabs_Ui_q[1][iquad].const_array();
        const auto &B0_qi = ec_fabs_Bi_ieside[0][(iquad == 0 || iquad == 3) ? 0 : 1].const_array();
        const auto &B1_qi = ec_fabs_Bi_ieside[1][(iquad < 2) ? 0 : 1].const_array();
        // compute electric field in the quadrant about the cell-edge: cross product between velocity and magnetic field in that
        // quadrant
        const auto &E2_qi = ec_fabs_E_q[iquad].array();
        amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
          E2_qi(i, j, k) = U0_qi(i, j, k) * B1_qi(i, j, k) - U1_qi(i, j, k) * B0_qi(i, j, k);
        });

        int tmp = 0; // TODO: for debuging. remove
      }

      // extract wavespeeds
      int w0_comp = -1;
      int w1_comp = -1;
      if (std::abs(w_extrap_dirs[0] - w_extrap_dirs[1]) == 2) {
        w0_comp = std::max(w_extrap_dirs[0], w_extrap_dirs[1]);
        w1_comp = std::min(w_extrap_dirs[0], w_extrap_dirs[1]);
      } else {
        w0_comp = std::min(w_extrap_dirs[0], w_extrap_dirs[1]);
        w1_comp = std::max(w_extrap_dirs[0], w_extrap_dirs[1]);
      }
      std::array<int, 3> delta_w0 = {0, 0, 0};
      std::array<int, 3> delta_w1 = {0, 0, 0};
      delta_w0[w0_comp] = 1;
      delta_w1[w1_comp] = 1;
      const auto &fspd_x0 = fcx_mf_fspds[w0_comp][mfi].const_array();
      const auto &fspd_x1 = fcx_mf_fspds[w1_comp][mfi].const_array();
      // extract both components of magnetic field either side of the cell-edge
      const auto &B0_m = ec_fabs_Bi_ieside[0][0].const_array();
      const auto &B0_p = ec_fabs_Bi_ieside[0][1].const_array();
      const auto &B1_m = ec_fabs_Bi_ieside[1][0].const_array();
      const auto &B1_p = ec_fabs_Bi_ieside[1][1].const_array();
      // extract all four quadrants of the electric field about the cell-edge
      const auto &E2_q0 = ec_fabs_E_q[0].const_array();
      const auto &E2_q1 = ec_fabs_E_q[1].const_array();
      const auto &E2_q2 = ec_fabs_E_q[2].const_array();
      const auto &E2_q3 = ec_fabs_E_q[3].const_array();
      // compute electric field on the cell-edge
      const auto &E2_ave = ec_mf_emf_comps[3-w0_comp-w1_comp][mfi].array();
      // only operate on the real cells
      amrex::ParallelFor(box_ec, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        const double fspd_x0_m = std::max(fspd_x0(i, j, k, 0), fspd_x0(i+delta_w0[0], j+delta_w0[1], k+delta_w0[2], 0));
        const double fspd_x0_p = std::max(fspd_x0(i, j, k, 1), fspd_x0(i+delta_w0[0], j+delta_w0[1], k+delta_w0[2], 1));
        const double fspd_x1_m = std::max(fspd_x1(i, j, k, 0), fspd_x1(i+delta_w1[0], j+delta_w1[1], k+delta_w1[2], 0));
        const double fspd_x1_p = std::max(fspd_x1(i, j, k, 1), fspd_x1(i+delta_w1[0], j+delta_w1[1], k+delta_w1[2], 1));
        const double E2_q0_ = E2_q0(i, j, k);
        const double E2_q1_ = E2_q1(i, j, k);
        const double E2_q2_ = E2_q2(i, j, k);
        const double E2_q3_ = E2_q3(i, j, k);
        const double B0_p_ = B0_p(i, j, k);
        const double B0_m_ = B0_m(i, j, k);
        const double B1_p_ = B1_p(i, j, k);
        const double B1_m_ = B1_m(i, j, k);
        const double _E2_ave = (
          (
            fspd_x0_p * fspd_x1_p * E2_q0_ +
            fspd_x0_p * fspd_x1_m * E2_q1_ +
            fspd_x0_m * fspd_x1_m * E2_q2_ +
            fspd_x0_m * fspd_x1_p * E2_q3_
          ) / (fspd_x0_m + fspd_x0_p) / (fspd_x1_m + fspd_x1_p) -
          fspd_x1_m * fspd_x1_p / (fspd_x1_m + fspd_x1_p) * (B0_p_ - B0_m_) +
          fspd_x0_m * fspd_x0_p / (fspd_x0_m + fspd_x0_p) * (B1_p_ - B1_m_)
        );
        E2_ave(i,j,k) = _E2_ave;
      });

      int tmp = 0; // TODO: for debuging. remove
		}
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::ReconstructTo(FluxDir dir, arrayconst_t &cState, array_t &lState, array_t &rState, const amrex::Box &box_cValid,
					 const int reconstructionOrder)
{
  amrex::Box const &box_r = amrex::grow(box_cValid, 1);
	amrex::Box const &box_r_x1 = amrex::surroundingNodes(box_r, static_cast<int>(dir));
  if (reconstructionOrder == 3) {
    // note: only box_r is used. box_r_x1 is unused.
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X1>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X2>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPPM<FluxDir::X3>(cState, lState, rState, box_r, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 2) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X1, SlopeLimiter::minmod>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X2, SlopeLimiter::minmod>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesPLM<FluxDir::X3, SlopeLimiter::minmod>(cState, lState, rState, box_r_x1, 1);
				break;
		}
	} else if (reconstructionOrder == 1) {
		switch (dir) {
			case FluxDir::X1:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X1>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X2:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X2>(cState, lState, rState, box_r_x1, 1);
				break;
			case FluxDir::X3:
				MHDSystem<problem_t>::template ReconstructStatesConstant<FluxDir::X3>(cState, lState, rState, box_r_x1, 1);
				break;
		}
	} else {
		amrex::Abort("Invalid reconstruction order specified!");
	}
}

template <typename problem_t>
void MHDSystem<problem_t>::SolveInductionEqn(std::array<amrex::MultiFab, AMREX_SPACEDIM> const &fc_consVarOld_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> &fc_consVarNew_mf, std::array<amrex::MultiFab, AMREX_SPACEDIM> const &ec_emf_mf, const double dt)
{
	// compute the total right-hand-side for the MOL integration

	// By convention, the fluxes are defined on the left edge of each zone,
	// i.e. flux_(i) is the flux *into* zone i through the interface on the
	// left of zone i, and -1.0*flux(i+1) is the flux *into* zone i through
	// the interface on the right of zone i.

  // loop over faces pointing in the w0-direction
  for (int w0 = 0; w0 < 3; ++w0) {
    // you have two edges on the perimeter of this face
    int w1 = (w0 + 1) % 3; // ivec_fc(w0) + ivec_fc(w1)
    int w2 = (w0 + 2) % 3; // ivec_fc(w0) + ivec_fc(w2)
    // let's figure out the direction we need to look up to find the respective edges either side of the face. this depends on the direction the face points
    std::array<int, 3> delta_w1 = {0, 0, 0};
    std::array<int, 3> delta_w2 = {0, 0, 0};
    if (w0 == 0) {
      delta_w1[1] = 1;
      delta_w2[2] = 1;
    } else if (w0 == 1) {
      delta_w1[2] = 1;
      delta_w2[0] = 1;
    } else if (w0 == 2) {
      delta_w1[0] = 1;
      delta_w2[1] = 1;
    }
    auto const ec_emf_w1 = ec_emf_mf[w1].const_arrays();
    auto const ec_emf_w2 = ec_emf_mf[w2].const_arrays();
    auto fc_consVarOld = fc_consVarOld_mf[w0].const_arrays();
    auto fc_consVarNew = fc_consVarNew_mf[w0].arrays();
    amrex::ParallelFor(fc_consVarNew_mf[w0], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
      // the ec emfs sit in the opposite fc directions relative to the face
      double emf_w1_m = ec_emf_w1[bx](i, j, k);
      double emf_w2_m = ec_emf_w2[bx](i, j, k);
      double emf_w1_p = ec_emf_w1[bx](i+delta_w2[0], j+delta_w2[1], k+delta_w2[2]);
      double emf_w2_p = ec_emf_w2[bx](i+delta_w1[0], j+delta_w1[1], k+delta_w1[2]);
      double db_dt = emf_w1_m - emf_w2_m - emf_w1_p + emf_w2_p;
      fc_consVarNew[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) = fc_consVarOld[bx](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex) + dt * db_dt;
    });

    int tmp = 0; // TODO: for debuging. remove 
  }
}

#endif // HYDRO_SYSTEM_HPP_
