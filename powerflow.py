import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

def initialize_powerflow(baseMVA, bus, gen, bus2index):

    # ---------------------------------------------------------
    # Bus Type
    # ---------------------------------------------------------

    bus_type = bus["BUS_TYPE"].to_numpy(dtype=int)

    bus_pq = np.where(bus_type == 1)[0]
    bus_pv = np.where(bus_type == 2)[0]
    bus_slack = np.where(bus_type == 3)[0]

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    Pd = bus["PD"].to_numpy(dtype=float) / baseMVA
    Qd = bus["QD"].to_numpy(dtype=float) / baseMVA

    # ---------------------------------------------------------
    # Online Generators
    # ---------------------------------------------------------
    gen_on = gen[gen["GEN_STATUS"]>0]

    gen_sum = (
        gen_on.groupby("GEN_BUS")[["PG", "QG"]]
        .sum()
        .reindex(bus2index, fill_value=0.0) # Reindex to match the index in internal ordering (Python), the bus with no generator will be filled with 0.0
    )

    Pg = gen_sum["PG"].to_numpy(dtype=float) / baseMVA
    Qg = gen_sum["QG"].to_numpy(dtype=float) / baseMVA

    # ---------------------------------------------------------
    # Specified Power
    # ---------------------------------------------------------

    P_spec = Pg - Pd
    Q_spec = Qg - Qd

    # ---------------------------------------------------------
    # Voltage Magnitude and Angle
    # ---------------------------------------------------------
    Vm = bus["VM"].to_numpy(dtype=float)
    Va = np.deg2rad(bus["VA"].to_numpy(dtype=float)) # Convert to radians   

    V = Vm * np.exp(1j * Va)

    return P_spec, Q_spec, Vm, Va, V, bus_slack, bus_pq, bus_pv

def calculate_power(Ybus, V):
    # Matrix multiplication to calculate current injections
    I = Ybus @ V
    # Element-wise multiplication to calculate complex power injections
    S = V * np.conj(I)

    P = S.real
    Q = S.imag
    return P, Q

def calculate_mismatch(
        P_spec, Q_spec,
        P_calc, Q_calc,
        bus_pv, bus_pq
):
    bus_pvpq = np.concatenate((bus_pv, bus_pq))

    dP = P_spec - P_calc
    dQ = Q_spec - Q_calc
    # [dP_pv, dP_pq, dQ_pq] = mismatch
    mismatch = np.concatenate((dP[bus_pvpq], dQ[bus_pq]))
    return mismatch

def build_jacobian(Ybus, Vm, Va, P, Q, bus_pv, bus_pq):

    nb = len(Vm)

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




def run_powerflow_newton(
        baseMVA, bus, gen, Ybus, bus2index,
        tol = 1e-8,
        max_iter = 10,
        flat_start = True
):
    """
    Solve AC power flow using the Newton-Raphson method.
    """

    # 1. Bus Classification
    bus_type = bus["BUS_TYPE"].to_numpy(dtype=int)

    bus_pq = np.where(bus_type == 1)[0]
    bus_pv = np.where(bus_type == 2)[0]
    bus_slack = np.where(bus_type == 3)[0]

    bus_pvpq = np.concatenate((bus_pv, bus_pq))
    npvpq = len(bus_pvpq)

    # 2. Load Power
    Pd = bus["PD"].to_numpy(dtype=float) / baseMVA
    Qd = bus["QD"].to_numpy(dtype=float) / baseMVA

    # 3. Generator Power
    gen_on = gen[gen["GEN_STATUS"]>0]

    gen_sum = (
        gen_on.groupby("GEN_BUS")[["PG", "QG"]]
        .sum()
        .reindex(bus2index, fill_value=0.0) # Reindex to match the index in internal ordering (Python), the bus with no generator will be filled with 0
    )

    Pg = gen_sum["PG"].to_numpy(dtype=float) / baseMVA
    Qg = gen_sum["QG"].to_numpy(dtype=float) / baseMVA

    # 4. Specified Power
    P_spec = Pg - Pd
    Q_spec = Qg - Qd

    # 5. Flat Start: Set all voltage magnitudes to 1.0 p.u. and angles to 0.0 radians
    nb = len(bus)
    Vm = np.ones(nb)
    Va = np.zeros(nb)
    Vm[bus_pv] = bus["VM"].to_numpy(dtype=float)[bus_pv]  # Set PV bus voltage magnitudes to specified values
    Vm[bus_slack] = bus["VM"].to_numpy(dtype=float)[bus_slack]  # Set slack bus voltage magnitude to specified value

    # 6. Iterative Newton-Raphson Method
    converged = False

    for iteration in range(max_iter+1):
        # 6.1. Construct the complex voltage vector V from Vm and Va
        V = Vm * np.exp(1j * Va)

        # 6.2. Calculate the power injection with current state variables
        P_calc, Q_calc = calculate_power(Ybus, V)

        # 6.3. Calculate the mismatch vector
        mismatch = calculate_mismatch(
            P_spec, Q_spec,
            P_calc, Q_calc,
            bus_pv, bus_pq
        )

        # 6.4. Check for convergence
        max_mismatch = np.max(np.abs(mismatch))
        print(f"Iteration {iteration + 1}: Max Mismatch = {max_mismatch:.6e}")
        if max_mismatch < tol:
            print("Power flow converged successfully.")
            converged = True
            break

        if iteration == max_iter:
            print("Maximum iterations reached without convergence.")
            break

        # 6.5. Build the Jacobian matrix
        J = build_jacobian(Ybus, Vm, Va, P_calc, Q_calc, bus_pv, bus_pq)

        # 6.6. Solve the Newton-Raphson update step
        dx = np.linalg.solve(J, mismatch) # J * dx = mismatch

        # 6.7. Update the state variables
        Va[bus_pvpq] += dx[:npvpq]  # Update voltage angles for PV and PQ buses
        Vm[bus_pq] += dx[npvpq:]  # Update voltage magnitudes for PQ buses

    # 7. Final Voltage (After Convergence or Max Iterations)
    
    V_final = Vm * np.exp(1j * Va)
    P_calc, Q_calc = calculate_power(Ybus, V_final)

    return converged, iteration, V_final, Vm, Va, P_calc, Q_calc

# def make_pq_matrices(Ybus, bus_pv, bus_pq):
#     """
#     Build fixed "Jacobian" Matrices for PQ decoupled power flow
#     """
#     bus_pvpq = np.concatenate((bus_pv,bus_pq))

#     B = -Ybus.imag

#     Bp = B[np.ix_(bus_pvpq, bus_pvpq)]
#     Bpp = B[np.ix_(bus_pq,bus_pq)]
#     return Bp, Bpp

def make_pq_matrices(bus, branch, bus_map, bus_pv, bus_pq):
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

    nb = len(bus)

    B = np.zeros((nb, nb))

    for _, line in branch.iterrows():

        if line["BR_STATUS"] == 0:
            continue

        from_bus = int(line["F_BUS"])
        to_bus = int(line["T_BUS"])

        i = bus_map[from_bus]
        j = bus_map[to_bus]

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

def run_PQDecoupled_powerflow(
    baseMVA, bus, branch, gen, Ybus, bus_ids,
    tol=1e-8,
    max_iter=100
):
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

    # --------------------------------------------------
    # 1. Bus classification
    # --------------------------------------------------

    bus_type = bus["BUS_TYPE"].to_numpy(dtype=int)

    pq = np.where(bus_type == 1)[0]
    pv = np.where(bus_type == 2)[0]
    slack = np.where(bus_type == 3)[0]

    pvpq = np.concatenate((pv, pq))

    # --------------------------------------------------
    # 2. Specified power injection
    # --------------------------------------------------

    Pd = bus["PD"].to_numpy(dtype=float) / baseMVA
    Qd = bus["QD"].to_numpy(dtype=float) / baseMVA

    gen_on = gen[gen["GEN_STATUS"] > 0]

    gen_sum = (
        gen_on
        .groupby("GEN_BUS")[["PG", "QG"]]
        .sum()
        .reindex(bus_ids, fill_value=0.0)
    )

    Pg = gen_sum["PG"].to_numpy(dtype=float) / baseMVA
    Qg = gen_sum["QG"].to_numpy(dtype=float) / baseMVA

    P_spec = Pg - Pd
    Q_spec = Qg - Qd

    # --------------------------------------------------
    # 3. Flat start
    # --------------------------------------------------

    nb = len(bus)

    Vm_set = bus["VM"].to_numpy(dtype=float)

    Vm = np.ones(nb)
    Va = np.zeros(nb)

    Vm[pv] = Vm_set[pv]
    Vm[slack] = Vm_set[slack]

    # --------------------------------------------------
    # 4. Build constant B' and B''
    # --------------------------------------------------

    Bp, Bpp = make_pq_matrices(
        bus,
        branch,
        bus_ids,
        pv,
        pq
    )

    # LU factorization
    Bp_lu, Bp_piv = lu_factor(Bp)
    Bpp_lu, Bpp_piv = lu_factor(Bpp)

    # --------------------------------------------------
    # 5. P-Q decoupled iteration
    # --------------------------------------------------

    converged = False

    for iteration in range(max_iter):

        # ==========================================
        # 1. P-theta correction
        # ==========================================

        V = Vm * np.exp(1j * Va)
        P_calc, Q_calc = calculate_power(Ybus, V)

        dP = P_spec - P_calc

        # B' * dTheta = dP/V
        # dVa = np.linalg.solve(Bp, dP[pvpq]/Vm[pvpq])
        # P*A = L*U
        rhs_P = dP[pvpq] / Vm[pvpq]
        dVa = lu_solve(
            (Bp_lu, Bp_piv),
            rhs_P
        )

        # Update theta
        Va[pvpq] += dVa

        # ==========================================
        # 2. Q-V correction
        # ==========================================

        # Recalculate Q after theta update
        V = Vm * np.exp(1j * Va)

        P_calc, Q_calc = calculate_power(Ybus, V)

        dQ = Q_spec - Q_calc

        #dVm = np.linalg.solve(Bpp, dQ[pq]/Vm[pq])
        rhs_Q = dQ[pq] / Vm[pq]
        dVm = lu_solve(
            (Bpp_lu, Bpp_piv),
            rhs_Q
        )

        # Update voltage magnitude
        Vm[pq] += dVm


        # ==========================================
        # 3. Convergence check
        # ==========================================

        # Recalculate P and Q using the NEW state
        V = Vm * np.exp(1j * Va)

        P_calc, Q_calc = calculate_power(Ybus, V)

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        max_dP = np.max(np.abs(dP[pvpq]))

        max_dQ = np.max(np.abs(dQ[pq]))

        max_mismatch = max(max_dP, max_dQ)

        print(
            f"Iteration {iteration + 1:2d}: "
            f"dP = {max_dP:.6e}, "
            f"dQ = {max_dQ:.6e}"
        )

        if max_mismatch < tol:
            converged = True
            break


    # --------------------------------------------------
    # 6. Final result
    # --------------------------------------------------

    V = Vm * np.exp(1j * Va)

    P_calc, Q_calc = calculate_power(Ybus, V)

    return (
        converged,
        iteration,
        Vm,
        Va,
        V,
        P_calc,
        Q_calc
    )