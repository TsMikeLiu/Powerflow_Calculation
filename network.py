import numpy as np
import pandas as pd

def create_bus_mapping(bus):
    """
    Create a mapping from bus IDs to indices and vice versa.

    Parameters
    ----------
    bus : pd.DataFrame
        DataFrame containing bus data with a column "BUS_I" for bus IDs.

    Returns
    -------
    bus2index : dict
        Dictionary mapping bus IDs to their corresponding indices.
    index2bus : dict
        Dictionary mapping indices to their corresponding bus IDs.
    """

    if bus["BUS_I"].duplicated().any():
        raise ValueError("Duplicate bus IDs found in the bus DataFrame.")

    bus_ids = bus["BUS_I"].astype(int).to_numpy()
    
    # {REAL_BUS_ID: INDEX}
    bus2index = {
        bus_id: idx
        for idx, bus_id in enumerate(bus_ids)
    }

    index2bus = {
        idx: bus_id
        for idx, bus_id in enumerate(bus_ids)
    }

    return bus2index, index2bus

def make_ybus(baseMVA, bus, branch):
    """
    Generate the Y Bus matrix for the power system.

    Parameters
    ----------
    baseMVA : float
        System base power in MVA.
    bus : pd.DataFrame
        DataFrame containing bus data.
    branch : pd.DataFrame
        DataFrame containing branch data.

    Returns
    -------
    Ybus : np.ndarray
        The Y Bus matrix as a complex numpy array.
    """

    nb = len(bus)
    bus2index, index2bus = create_bus_mapping(bus)
    Ybus = np.zeros(
        (nb, nb),
        dtype=complex
    )

    # ----------------------
    # Branch Admittance
    # ----------------------

    for _,line in branch.iterrows():
        # Skip out-of-service branches
        if line["BR_STATUS"] == 0:
            continue

        # Bus Index in External World (Real Bus ID)
        from_bus = int(line["F_BUS"])
        to_bus = int(line["T_BUS"])

        # Bus Index in Python (0-based)
        i,j = bus2index[from_bus], bus2index[to_bus]

        # The value here is based on p.u.
        r = line["BR_R"]
        x = line["BR_X"]
        b = line["BR_B"]

        # Serires Admittance
        z = r + 1j*x
        y = 1/z

        # Line charging susceptance
        y_shunt = 1j*b/2

        # Transformer tap ratio
        tap = line["TAP"]

        # regulation the rule in Matpower, if tap is 0, it means no transformer, so we set it to transformer tap ratio to 1.0
        if tap == 0:
            tap = 1.0

        # Phase shift: from degree to radian
        shift = np.deg2rad(line["SHIFT"])

        # Complex tap ratio
        tap = tap * np.exp(1j*shift)

        # Branch Admittance Matrix Contribution
        Yff = (y + y_shunt) / (abs(tap)**2)
        Yft = -y / np.conj(tap)
        Ytf = -y / tap
        Ytt = y + y_shunt

        Ybus[i,i] += Yff
        Ybus[j,j] += Ytt
        Ybus[i,j] += Yft
        Ybus[j,i] += Ytf

    for _, row in bus.iterrows():

        bus_id = int(row["BUS_I"])
        i = bus2index[bus_id]

        # The value here is based on MW / MVAr, not p.u.
        gs = row["GS"]
        bs = row["BS"]

        Ybus[i, i] += (
            gs + 1j * bs
        ) / baseMVA

    # print(Ybus)
    return Ybus