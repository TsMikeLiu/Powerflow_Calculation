import numpy as np
import pandas as pd
from data.case14 import case14
from network import PowerSystem
# from powerflow import calculate_power, calculate_mismatch, run_powerflow_newton, run_PQDecoupled_powerflow
from powerflow import NewtonRaphsonSolver, PQDecoupledSolver, PowerFlowSolver

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
# converged, iteration, Vm, Va, V_final, P_calc, Q_calc = run_PQDecoupled_powerflow(
#         system,
#         tol = 1e-8,
#         max_iter = 50,
#         )
# if converged:
#     print(f"Power flow converged in {iteration + 1} iterations.")
# else:
#     print(f"Power flow did not converge in {max_iter} iterations.") 

# Va_deg = np.rad2deg(Va)
# result = pd.DataFrame(
#     {
#         "BUS_I": bus["BUS_I"],
#         "VM": Vm,
#         "VA": Va_deg,
#         "P_inj": P_calc * baseMVA,
#         "Q_inj": Q_calc * baseMVA
#     }
# )

# print()
# print(result)







# bus = pd.DataFrame(bus_data, columns=bus_columns)

# print(bus)
# print(type(bus))

# DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects. It is generally the most commonly used pandas object.
# Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index. A Series is like a fixed-size dict in that you can get and set values by index label.
# PD = bus["PD"]
# print(PD)
# print(type(PD))

# PD = bus["PD"].to_numpy()
# print(PD)
# print(type(PD))

# slack_bus_id = bus.loc[bus["BUS_TYPE"] == 3, "BUS_I"].to_numpy()
# print(slack_bus_id)