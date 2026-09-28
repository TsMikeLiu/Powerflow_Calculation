import numpy as np
import pandas as pd
from data.case14 import case14
from network import create_bus_mapping, make_ybus
from powerflow import initialize_powerflow, calculate_power, calculate_mismatch, build_jacobian, run_powerflow_newton, run_PQDecoupled_powerflow

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

# print(bus.head())
# print(gen.head())
# print(branch.head())
bus2index, index2bus = create_bus_mapping(bus)
# Generate Y Bus
Ybus = make_ybus(baseMVA, bus, branch)
# np.set_printoptions(
#     precision = 4,
#     suppress = True
# )
# print(Ybus)

converged, iteration, V_final, Vm, Va, P_calc, Q_calc = run_powerflow_newton(
        baseMVA, bus, gen, Ybus, bus2index,
        tol = 1e-8,
        max_iter = 10,
        flat_start = True
        )

Va_deg = np.rad2deg(Va)  # Convert voltage angles from radians to degrees
print()




if converged:
    print(f"Power flow converged in {iteration + 1} iterations.")
else:
    print(f"Power flow did not converge in {max_iter} iterations.") 

result = pd.DataFrame(
    {
        "BUS_I": bus["BUS_I"],
        "VM": Vm,
        "VA": Va_deg,
        "P_inj": P_calc * baseMVA,
        "Q_inj": Q_calc * baseMVA
    }
)

print()
print(result)


converged, iteration, Vm, Va, V_final, P_calc, Q_calc = run_PQDecoupled_powerflow(
        baseMVA, bus, branch, gen, Ybus, bus2index,
        tol = 1e-8,
        max_iter = 50,
        )
if converged:
    print(f"Power flow converged in {iteration + 1} iterations.")
else:
    print(f"Power flow did not converge in {max_iter} iterations.") 

Va_deg = np.rad2deg(Va)
result = pd.DataFrame(
    {
        "BUS_I": bus["BUS_I"],
        "VM": Vm,
        "VA": Va_deg,
        "P_inj": P_calc * baseMVA,
        "Q_inj": Q_calc * baseMVA
    }
)

print()
print(result)







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