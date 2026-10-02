import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve



class PowerFlowSolver:
    def __init__(
            self,
            system,
            tol = 1e-8,
            max_iter = 20,
            flat_start = True
    ):
        """
        Base Class for Power Flow Solver
        """
        self.system = system
        self.tol = tol
        self.max_iter = max_iter
        self.flat_start = flat_start

        # Initialize Voltage State
        self.Vm, self.Va = self.initialize_voltage()

        # Solver status
        self.converged = False
        self.iteration = 0

        # Solver result
        self.V = None
        self.P_calc = None
        self.Q_calc = None


    def initialize_voltage(self):
        """
        Initialize system voltage magnitude and angle
        """
        system = self.system

        if self.flat_start:
            Vm = np.ones(system.nb)
            Va = np.zeros(system.nb)

            Vm_set = system.bus["VM"].to_numpy(dtype=float)

            Vm[system.bus_pv] = Vm_set[system.bus_pv]
            Vm[system.bus_slack] = Vm_set[system.bus_slack]

        else:
            # Here we use copy, because we need to regulate the value of Vm
            Vm = system.bus["VM"].to_numpy(dtype=float).copy()
            Va = np.deg2rad(
                system.bus["VA"].to_numpy(dtype=float)
            )
        
        return Vm, Va

    def get_complex_voltage(self):
        """
        Construst complex bus voltage vector
        """
        return self.Vm * np.exp(1j * self.Va)

    def calculate_power(self):
        """
        Calculate bus active and reactive power injection
        """
        V = self.get_complex_voltage()

        I = self.system.Ybus @ V

        S = V * np.conj(I)

        P = S.real
        Q = S.imag

        return P, Q

    def calculate_mismatch(self, P_calc, Q_calc):
        """
        Calculate the mismatch of power
        """
        system = self.system

        dP = system.P_spec - P_calc
        dQ = system.Q_spec - Q_calc

        return dP, dQ

class NewtonRaphsonSolver(PowerFlowSolver):
    def __init__(self, system, tol=1e-8, max_iter=20, flat_start=True):
        super().__init__(system, tol, max_iter, flat_start)

    def build_jacobian(self, P, Q):

        system = self.system

        Ybus = system.Ybus
        Vm = self.Vm
        Va = self.Va

        nb = system.nb

        bus_pv = system.bus_pv
        bus_pq = system.bus_pq
        bus_pvpq = system.bus_pvpq
        bus_slack = system.bus_slack
        
        G = Ybus.real
        B = Ybus.imag
    
        J11 = np.zeros((nb, nb))
        J12 = np.zeros((nb, nb))
        J21 = np.zeros((nb, nb))
        J22 = np.zeros((nb, nb))
    
        for i in range(nb):
            for k in range(nb):
                if i == k:
                    J11[i, k] = -Q[i] - B[i, i] * Vm[i] ** 2
                    J12[i, k] = P[i] / Vm[i] + G[i, i] * Vm[i]
                    J21[i, k] = P[i] - G[i, i] * Vm[i] ** 2
                    J22[i, k] = Q[i] / Vm[i] - B[i, i] * Vm[i]
                else:
                    theta_ik = Va[i] - Va[k]
                    J11[i, k] = Vm[i] * Vm[k] * (G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik))
                    J12[i, k] = Vm[i] * (G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik))
                    J21[i, k] = - Vm[i] * Vm[k] * (G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik))
                    J22[i, k] = Vm[i] * (G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik))
    
        # Pickup the submatrices corresponding to PV and PQ buses
        bus_pvpq = np.concatenate((bus_pv, bus_pq))
        # Num of variable theta Va:     PV + PQ buses
        # Num of variable voltage Vm:   PQ buses
        J11 = J11[np.ix_(bus_pvpq, bus_pvpq)]   # derivative of P w.r.t. Va for PQ and PV buses
        J12 = J12[np.ix_(bus_pvpq, bus_pq)]     # derivative of P w.r.t. Vm for PQ buses (only OQ buses have voltage magnitude as variable)
        J21 = J21[np.ix_(bus_pq, bus_pvpq)]     # derivative of Q w.r.t. Va for PQ and PV buses
        J22 = J22[np.ix_(bus_pq, bus_pq)]       # derivative of Q w.r.t. Vm for PQ buses (only OQ buses have voltage magnitude as variable)
    
        J = np.block([[J11, J12],
                        [J21, J22]])
        return J

    def solve(self):
        system = self.system

        converged = False
        npvpq = len(system.bus_pvpq)

        print(f"==============================================================")
        print(f"Power Flow Calculation Started, Using Newton Raphson Method")
        print(f"==============================================================")
        for iteration in range(self.max_iter+1):

            P_calc, Q_calc = self.calculate_power()
    
            dP, dQ = self.calculate_mismatch(P_calc, Q_calc)

            mismatch = np.concatenate(
                (dP[system.bus_pvpq], dQ[system.bus_pq])
            )

            max_mismatch = np.max(np.abs(mismatch))
            print(f"Iteration {iteration + 1}: Max Mismatch = {max_mismatch:.6e}")
            if max_mismatch < self.tol:
                print("Power flow converged successfully.")
                converged = True
                break
    
            if iteration == self.max_iter:
                print("Maximum iterations reached without convergence.")
                break
    
            J = self.build_jacobian(P_calc, Q_calc)
    
            dx = np.linalg.solve(J, mismatch) # J * dx = mismatch

            self.Va[system.bus_pvpq] += dx[:npvpq]  # Update voltage angles for PV and PQ buses
            self.Vm[system.bus_pq] += dx[npvpq:]  # Update voltage magnitudes for PQ buses

        print(f"==============================================================")
        print(f"Power Flow Calculation Ended, Using Newton Raphson Method")
        print(f"==============================================================")

        # 7. Final Voltage (After Convergence or Max Iterations)
        
        self.converged = converged
        self.iteration = iteration

        self.V = self.get_complex_voltage()
        self.P_calc, self.Q_calc = self.calculate_power()

        return self.converged

class PQDecoupledSolver(PowerFlowSolver):
    def __init__(self, system, tol=1e-8, max_iter=20, flat_start=True):
        super().__init__(system, tol, max_iter, flat_start)

        # Build B' and B'' Matrices
        self.Bp, self.Bpp = self.make_pq_matrices()

        # LU factorization: Only compute once
        self.Bp_lu, self.Bp_piv = lu_factor(self.Bp)
        self.Bpp_lu, self.Bpp_piv = lu_factor(self.Bpp)

    def make_pq_matrices(self):
        """
        Build fixed B' and B'' matrices for P-Q decoupled power flow.

        Approximation:
            R << X
            small angle differences
            Vm ≈ 1 p.u.

        Ignore:
            branch resistance
            line charging
            bus shunt
            phase shift

        Keep:
            branch reactance
            transformer tap ratio
        """
        system = self.system
        bus = system.bus
        branch = system.branch
        nb = system.nb
        bus2index = system.bus2index
        bus_pv = system.bus_pv
        bus_pq = system.bus_pq
        bus_pvpq = system.bus_pvpq

        B = np.zeros((nb, nb))

        for _, line in branch.iterrows():

            if line["BR_STATUS"] == 0:
                continue

            from_bus = int(line["F_BUS"])
            to_bus = int(line["T_BUS"])

            i = bus2index[from_bus]
            j = bus2index[to_bus]

            x = line["BR_X"]

            tap = line["TAP"]

            if tap == 0:
                tap = 1.0

            # Series susceptance magnitude
            b = 1.0 / x

            # Transformer-adjusted branch contributions
            B[i, i] += b / (tap ** 2)
            B[i, j] -= b / tap

            B[j, i] -= b / tap
            B[j, j] += b

        bus_pvpq = np.concatenate((bus_pv, bus_pq))

        Bp = B[np.ix_(bus_pvpq, bus_pvpq)]
        Bpp = B[np.ix_(bus_pq, bus_pq)]

        return Bp, Bpp

    def solve(self):
        """
        Solve AC power flow using the P-Q decoupled method.
        Old Vm, Va
    │
    ├── calculate ΔP
    │
    ├── solve B' Δθ = ΔP / V
    │
    └── update θ
            │
            ▼
        New θ，Old Vm
            │
            ├── recalculate ΔQ
            │
            ├── solve B'' ΔV = ΔQ / V
            │
            └── update Vm
                        │
                        ▼
                New θ，New Vm
                        │
                        ├── recalculate ΔP
                        ├── recalculate ΔQ
                        │
                        ▼
                convergence check
        """
        system = self.system
        self.converged = False
        print(f"==============================================================")
        print(f"Power Flow Calculation Started, Using PQ Decoupled Method")
        print(f"==============================================================")
        for iteration in range(self.max_iter):
            
            P_calc, Q_calc = self.calculate_power()

            dP, dQ = self.calculate_mismatch(P_calc, Q_calc)

            # ===========================================================
            # 1. P-Va Correciton
            # ===========================================================
            rhs_P = dP[system.bus_pvpq] / self.Vm[system.bus_pvpq]

            dVa = lu_solve(
                (self.Bp_lu, self.Bp_piv),
                rhs_P
            )

            self.Va[system.bus_pvpq] += dVa

            # ===========================================================
            # 2. Q-Vm Correction
            # ===========================================================

            # Re-compute the mismatch of P and Q
            P_calc, Q_calc = self.calculate_power()
            dP, dQ = self.calculate_mismatch(P_calc, Q_calc)

            rhs_Q = dQ[system.bus_pq] / self.Vm[system.bus_pq]

            dVm = lu_solve(
                (self.Bpp_lu, self.Bpp_piv),
                rhs_Q
            )

            self.Vm[system.bus_pq] += dVm

            # ===========================================================
            # 3. Converge Check
            # ===========================================================

            P_calc, Q_calc = self.calculate_power()
            dP, dQ = self.calculate_mismatch(P_calc, Q_calc)

            max_mismatch = np.max(np.abs(
                np.concatenate(
                    (dP[system.bus_pvpq], dQ[system.bus_pq])
                )
            ))

            print(f"Iteration {iteration + 1:2d}: "
                  f"max_mismatch = {max_mismatch:.6e}")

            if max_mismatch < self.tol:
                self.converged = True
                break

        # Final Calculation and Store the results

        self.iteration = iteration
        self.V = self.get_complex_voltage()
        self.P_calc, self.Q_calc = self.calculate_power()
        print(f"==============================================================")
        print(f"Power Flow Calculation Ended, Using PQ Decoupled Method")
        print(f"==============================================================")
        return self.converged

class DLPFSolver(PowerFlowSolver):
    """
    Decoupled Linear Power Flow
    The Power flow equaiton is totally linearized. Thus the DLPF does not need iterative calculation
    Origin: Jingwei Yang
    """
    def __init__(self, system):
        super().__init__(
            system, 
            tol = 0.0, 
            max_iter = 1, 
            flat_start=False
            )
        

    def build_dlpf_matrices(self):
        """
        Build four basic matrices for DLPF
        [ Pm1 ]   [ B11  G12 ] [ theta ]
        [ Qm1 ] = [ G21  B22 ] [   V   ]
        """
        system = self.system

        Gbus = system.Ybus.real
        Bbus = system.Ybus.imag

        GP = Gbus

        # Shunt contribution
        BD = np.diag(
            np.sum(Bbus, axis=0)
        )

        # remove the shunt contribution
        BP = -Bbus + BD
        BQ = -Bbus
        GQ = -Gbus

        B11 = BP[np.ix_(system.bus_pvpq, system.bus_pvpq)]
        G12 = GP[np.ix_(system.bus_pvpq, system.bus_pq)]
        G21 = GQ[np.ix_(system.bus_pq, system.bus_pvpq)]
        B22 = BQ[np.ix_(system.bus_pq, system.bus_pq)]

        return B11, G12, G21, B22, BD

    def build_modified_injections(self):
        """
        Build modified active and reactive power injections
        for the DLPF model.

        Unknown states:
            theta -> PV and PQ bus voltage angles
            V     -> PQ bus voltage magnitudes
        """

        system = self.system

        # Full admittance matrix
        Gbus = system.Ybus.real
        Bbus = system.Ybus.imag

        # Matrices in Eq. (8)
        GP = Gbus
        BQ = -Bbus

        bus_slack = system.bus_slack
        bus_pv = system.bus_pv
        bus_pq = system.bus_pq
        bus_pvpq = system.bus_pvpq

        # Known voltage magnitudes
        V_ref = system.bus["VM"].to_numpy(dtype=float)[bus_slack]
        V_pv = system.bus["VM"].to_numpy(dtype=float)[bus_pv]

        # Modified active power injection
        Pm1 = (
            system.P_spec[bus_pvpq]
            - GP[np.ix_(bus_pvpq, bus_slack)] @ V_ref
            - GP[np.ix_(bus_pvpq, bus_pv)] @ V_pv
        )

        # Modified reactive power injection
        Qm1 = (
            system.Q_spec[bus_pq]
            - BQ[np.ix_(bus_pq, bus_slack)] @ V_ref
            - BQ[np.ix_(bus_pq, bus_pv)] @ V_pv
        )

        return Pm1, Qm1

    def build_decoupled_matices(self):
        """
        Build the decoupled DLPF systems:

            B11m @ theta = Pm2
            B22m @ V     = Qm2

        using Schur complements.
        """

        # --------------------------------------------------
        # t1 = G12 / B22 = G12 @ inv(B22)
        # --------------------------------------------------

        # Avoid explicit matrix inverse.
        # Solve:
        #     B22.T @ t1.T = G12.T
        t1 = np.linalg.solve(
            self.B22.T,
            self.G12.T
        ).T

        # --------------------------------------------------
        # t2 = G21 / B11 = G21 @ inv(B11)
        # --------------------------------------------------

        t2 = np.linalg.solve(
            self.B11.T,
            self.G21.T
        ).T

        # --------------------------------------------------
        # Schur complement matrices
        # --------------------------------------------------

        B11m = (
            self.B11
            - t1 @ self.G21
        )

        B22m = (
            self.B22
            - t2 @ self.G12
        )

        # # --------------------------------------------------
        # # Modified power injections
        # # --------------------------------------------------

        # Pm2 = (
        #     self.Pm1
        #     - t1 @ self.Qm1
        # )

        # Qm2 = (
        #     self.Qm1
        #     - t2 @ self.Pm1
        # )

        return B11m, B22m

    def build_decoupled_rhs(self):
        """
        Build Pm2 and Qm2.
        """

        B22_inv_Qm1 = np.linalg.solve(
            self.B22,
            self.Qm1
        )

        B11_inv_Pm1 = np.linalg.solve(
            self.B11,
            self.Pm1
        )

        Pm2 = (
            self.Pm1
            - self.G12 @ B22_inv_Qm1
        )

        Qm2 = (
            self.Qm1
            - self.G21 @ B11_inv_Pm1
        )

        return Pm2, Qm2

    def recover_voltage(self, theta, V_pq):
        """
        Recover complete Vm and Va vectors.
        """

        system = self.system

        Vm_set = system.bus["VM"].to_numpy(
            dtype=float
        )

        self.Vm[system.bus_slack] = Vm_set[system.bus_slack]

        self.Vm[system.bus_pv] = Vm_set[system.bus_pv]

        self.Vm[system.bus_pq] = V_pq

        Va_ref = np.deg2rad(
            system.bus["VA"]
            .to_numpy(dtype=float)[system.bus_slack]
        )[0]

        self.Va[:] = Va_ref

        self.Va[system.bus_pvpq] += theta

    def solve(self):
        """
        Solve the Decoupled Linearized Power Flow (DLPF)
        """

        system = self.system

        # Build Matrices
        self.B11, self.G12, self.G21, self.B22, self.BD = self.build_dlpf_matrices()

        # Calculate the modified power injection
        self.Pm1, self.Qm1 = self.build_modified_injections()

        # Calculate the decoupled matrices
        self.B11m, self.B22m = self.build_decoupled_matices()

        self.Pm2, self.Qm2 = self.build_decoupled_rhs()


        # ------------------------------------------
        # Solve voltage phase angles
        # B11m @ theta = Pm2
        # ------------------------------------------
        theta = np.linalg.solve(self.B11m, self.Pm2)

        # ------------------------------------------
        # Solve PQ-bus voltage magnitudes
        # B22m @ V = Qm2
        # ------------------------------------------
        V_pq = np.linalg.solve(
            self.B22m,
            self.Qm2
        )

        #  Recover the voltage vector
        self.recover_voltage(theta, V_pq)

        # ------------------------------------------
        # Store complex voltage
        # ------------------------------------------

        self.V = self.get_complex_voltage()

        # Evaluate the DLPF solution using full AC equations
        self.P_calc, self.Q_calc = self.calculate_power()

        self.converged = True

        return self.converged


class FDLPFSolver(DLPFSolver):

    def __init__(self, system):

        super().__init__(system)

        self.B_fast = None
        self.Pbusinj = None
        self.Qbusinj = None
        self.Pfinj = None


    def make_xb(self):
        """
        Build B matrix and phase-shift injection vectors
        for FDLPF.

        Python implementation of the author's makeXB.m.

        Returns
        -------
        Bbus : ndarray, shape (nb, nb)
            Fast B matrix based on 1 / (x * tap).

        Pbusinj : ndarray, shape (nb,)
            Active-power bus injection caused by
            phase shifters.

        Qbusinj : ndarray, shape (nb,)
            Reactive-power bus injection caused by
            phase shifters.

        Pfinj : ndarray, shape (nl,)
            Active-power branch-flow injection caused by
            phase shifters.
        """

        system = self.system

        nb = system.nb
        nl = len(system.branch)

        Bbus = np.zeros(
            (nb, nb),
            dtype=float
        )

        Pbusinj = np.zeros(nb)
        Qbusinj = np.zeros(nb)
        Pfinj = np.zeros(nl)

        for k, (_, line) in enumerate(
            system.branch.iterrows()
        ):

            if line["BR_STATUS"] == 0:
                continue

            # --------------------------------------
            # Bus indices
            # --------------------------------------

            from_bus = int(line["F_BUS"])
            to_bus = int(line["T_BUS"])

            i = system.bus2index[from_bus]
            j = system.bus2index[to_bus]

            # --------------------------------------
            # Branch parameters
            # --------------------------------------

            r = line["BR_R"]
            x = line["BR_X"]

            tap = line["TAP"]

            if tap == 0:
                tap = 1.0

            shift = np.deg2rad(
                line["SHIFT"]
            )

            # --------------------------------------
            # Fast B matrix
            #
            # MATLAB:
            #
            # b = stat ./ BR_X;
            # b = b ./ tap;
            #
            # Bbus = Cft' * Bf
            # --------------------------------------

            b = 1.0 / (x * tap)

            Bbus[i, i] += b
            Bbus[i, j] -= b

            Bbus[j, i] -= b
            Bbus[j, j] += b

            # --------------------------------------
            # Phase-shift injections
            # --------------------------------------

            denominator = (
                r ** 2
                + x ** 2
            )

            bij = (
                -x
                / denominator
                / tap
            )

            gij = (
                r
                / denominator
                / tap
            )

            # MATLAB:
            #
            # Pfinj =
            # -bij .* (-SHIFT*pi/180)
            #
            # Qfinj =
            # -gij .* (-SHIFT*pi/180)

            Pfinj_k = bij * shift
            Qfinj_k = gij * shift

            Pfinj[k] = Pfinj_k

            # Equivalent to:
            #
            # Pbusinj = Cft.T @ Pfinj
            # Qbusinj = Cft.T @ Qfinj

            Pbusinj[i] += Pfinj_k
            Pbusinj[j] -= Pfinj_k

            Qbusinj[i] += Qfinj_k
            Qbusinj[j] -= Qfinj_k

        return (
            Bbus,
            Pbusinj,
            Qbusinj,
            Pfinj
        )


    def build_fast_decoupled_matrices(self):
        """
        Build the accelerated B11m and B22m matrices
        used by FDLPF.
        """

        system = self.system

        bus_pvpq = system.bus_pvpq
        bus_pq = system.bus_pq

        B11m = self.B_fast[
            np.ix_(bus_pvpq, bus_pvpq)
        ]

        B22m = (
            self.B_fast[
                np.ix_(bus_pq, bus_pq)
            ]
            - self.BD[
                np.ix_(bus_pq, bus_pq)
            ]
        )

        return B11m, B22m


    def solve(self):
        """
        Solve Fast Decoupled Linearized Power Flow.
        """

        system = self.system

        # ==================================================
        # Step 1
        # Build Ybus with phase shifts temporarily removed
        #
        # MATLAB:
        # mpc.branch(:, SHIFT) = 0;
        # Ybus = makeYbus(mpc);
        # ==================================================

        Ybus = system.make_ybus(
            ignore_shift=True
        )

        # ==================================================
        # Step 2
        # Build basic DLPF matrices
        #
        # B11, G12, G21, B22
        # ==================================================

        (
            self.B11,
            self.G12,
            self.G21,
            self.B22,
            self.BD
        ) = self.build_dlpf_matrices()

        # ==================================================
        # Step 3
        # Build Pm1 and Qm1
        # ==================================================

        (
            self.Pm1,
            self.Qm1
        ) = self.build_modified_injections()

        # ==================================================
        # Step 4
        # FDLPF acceleration matrix
        # and phase-shift injections
        # ==================================================

        (
            self.B_fast,
            self.Pbusinj,
            self.Qbusinj,
            self.Pfinj
        ) = self.make_xb()

        # ==================================================
        # Step 5
        # Build approximate B11m and B22m
        # ==================================================

        (
            self.B11m,
            self.B22m
        ) = self.build_fast_decoupled_matrices()

        # ==================================================
        # Step 6
        # Correct Pm1 and Qm1 for phase shifters
        #
        # MATLAB:
        #
        # Pm1 = Pm1 - Pbusinj([bus_pv;bus_pq]);
        # Qm1 = Qm1 - Qbusinj(bus_pq);
        # ==================================================

        self.Pm1 = (
            self.Pm1
            - self.Pbusinj[
                system.bus_pvpq
            ]
        )

        self.Qm1 = (
            self.Qm1
            - self.Qbusinj[
                system.bus_pq
            ]
        )

        # ==================================================
        # Step 7
        # Build Pm2 and Qm2
        #
        # Same equations as DLPF
        # ==================================================

        (
            self.Pm2,
            self.Qm2
        ) = self.build_decoupled_rhs()

        # ==================================================
        # Step 8
        # Direct linear solve
        # ==================================================

        theta = np.linalg.solve(
            self.B11m,
            self.Pm2
        )

        V_pq = np.linalg.solve(
            self.B22m,
            self.Qm2
        )

        # ==================================================
        # Step 9
        # Recover complete voltage state
        # ==================================================

        self.recover_voltage(
            theta,
            V_pq
        )

        self.V = self.get_complex_voltage()

        # Full AC power equations are used only
        # to evaluate the approximation error.
        self.P_calc, self.Q_calc = \
            self.calculate_power()

        self.converged = True

        return self.converged