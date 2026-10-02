import numpy as np
import pandas as pd
from data.case14 import case14
from network import PowerSystem
# from powerflow import calculate_power, calculate_mismatch, run_powerflow_newton, run_PQDecoupled_powerflow
from powerflow import NewtonRaphsonSolver, PQDecoupledSolver, PowerFlowSolver, DLPFSolver, FDLPFSolver

flat_start = True
max_iter = 10
tol = 1e-8


baseMVA, bus_data, gen_data, branch_data = case14()


bus_columns = [
    "BUS_I", "BUS_TYPE", "PD", "QD",
    "GS", "BS", "BUS_AREA",
    "VM", "VA", "BASE_KV",
    "ZONE", "VMAX", "VMIN"
]
gen_columns = [
    "GEN_BUS",
    "PG",
    "QG",
    "QMAX",
    "QMIN",
    "VG",
    "MBASE",
    "GEN_STATUS",
    "PMAX",
    "PMIN",
    "PC1",
    "PC2",
    "QC1MIN",
    "QC1MAX",
    "QC2MIN",
    "QC2MAX",
    "RAMP_AGC",
    "RAMP_10",
    "RAMP_30",
    "RAMP_Q",
    "APF"
]
branch_columns = [
    "F_BUS",
    "T_BUS",
    "BR_R",
    "BR_X",
    "BR_B",
    "RATE_A",
    "RATE_B",
    "RATE_C",
    "TAP",
    "SHIFT",
    "BR_STATUS",
    "ANGMIN",
    "ANGMAX"
]

bus = pd.DataFrame(bus_data, columns=bus_columns)
gen = pd.DataFrame(gen_data, columns=gen_columns)
branch = pd.DataFrame(branch_data, columns=branch_columns)

system = PowerSystem(
    baseMVA,
    bus,
    gen,
    branch
)

nr_solver = NewtonRaphsonSolver(
    system,
    tol = 1e-8,
    max_iter = 20,
    flat_start = True
)
nr_solver.solve()


Va_deg = np.rad2deg(nr_solver.Va)  # Convert voltage angles from radians to degrees
print()

if nr_solver.converged:
    print(f"Power flow converged in {nr_solver.iteration + 1} iterations.")
else:
    print(f"Power flow did not converge in {nr_solver.max_iter} iterations.") 

result = pd.DataFrame(
    {
        "BUS_I": bus["BUS_I"],
        "VM": nr_solver.Vm,
        "VA": Va_deg,
        "P_inj": nr_solver.P_calc * baseMVA,
        "Q_inj": nr_solver.Q_calc * baseMVA
    }
)

print()
print(result)

pq_solver = PQDecoupledSolver(
    system,
    tol=1e-8,
    max_iter=50,
    flat_start=True
)

pq_solver.solve()

Va_deg = np.rad2deg(pq_solver.Va)  # Convert voltage angles from radians to degrees
print()

if pq_solver.converged:
    print(f"Power flow converged in {pq_solver.iteration + 1} iterations.")
else:
    print(f"Power flow did not converge in {pq_solver.max_iter} iterations.") 

result = pd.DataFrame(
    {
        "BUS_I": bus["BUS_I"],
        "VM": pq_solver.Vm,
        "VA": Va_deg,
        "P_inj": pq_solver.P_calc * baseMVA,
        "Q_inj": pq_solver.Q_calc * baseMVA
    }
)
print()
print(result)

dlpf_solver = DLPFSolver(system)
dlpf_solver.solve()

fdlpf_solver = FDLPFSolver(system)
fdlpf_solver.solve()

comparison = pd.DataFrame({
    "BUS_I": bus["BUS_I"],
    "BUS_TYPE": bus["BUS_TYPE"],

    "VM_NR": nr_solver.Vm,
    "VM_DLPF": dlpf_solver.Vm,
    "VM_FDLPF": fdlpf_solver.Vm,

    "VA_NR": np.rad2deg(
        nr_solver.Va
    ),

    "VA_DLPF": np.rad2deg(
        dlpf_solver.Va
    ),

    "VA_FDLPF": np.rad2deg(
            fdlpf_solver.Va
        )
})

comparison["VM_Error_DLPF"] = (
    comparison["VM_DLPF"]
    - comparison["VM_NR"]
)

comparison["VM_Error_FDLPF"] = (
    comparison["VM_FDLPF"]
    - comparison["VM_NR"]
)

comparison["VA_Error_deg_DLPF"] = (
    comparison["VA_DLPF"]
    - comparison["VA_NR"]
)

comparison["VA_Error_deg_FDLPF"] = (
    comparison["VA_FDLPF"]
    - comparison["VA_NR"]
)

print(comparison)