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



