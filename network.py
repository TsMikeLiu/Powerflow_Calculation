import numpy as np
import pandas as pd
    

class PowerSystem:
    def __init__(self, baseMVA, bus, gen, branch):
        self.baseMVA = baseMVA
        self.bus = bus
        self.gen = gen
        self.branch = branch

        # Basic System Size
        self.nb = len(bus)

        # Internal index & External index Mapping
        self.bus2index, self.index2bus = self.create_bus_mapping()

        # Bus Classification
        self.bus_pq, self.bus_pv, self.bus_slack, self.bus_pvpq = self.classify_buses()

        # Calculate Specified Power Injection
        self.P_spec, self.Q_spec = self.calculate_specified_power()

        # Calculate Y Bus
        self.Ybus = self.make_ybus()


    def create_bus_mapping(self):
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
    
        if self.bus["BUS_I"].duplicated().any():
            raise ValueError("Duplicate bus IDs found in the bus DataFrame.")
    
        bus_ids = self.bus["BUS_I"].astype(int).to_numpy()
        
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

    def classify_buses(self):
        """
        Classify buses according to MATPOWER bus types
        """
        bus_type = self.bus["BUS_TYPE"].to_numpy(dtype=int)

        bus_pq = np.where(bus_type == 1)[0]
        bus_pv = np.where(bus_type == 2)[0]
        bus_slack = np.where(bus_type == 3)[0]

        bus_pvpq = np.concatenate(
            (bus_pv,bus_pq)
        )

        return bus_pq, bus_pv, bus_slack, bus_pvpq
        
    def calculate_specified_power(self):
        """
        Calculate specified NET P & Q injection
        """
        # Load demand
        Pd = self.bus["PD"].to_numpy(dtype=float) / self.baseMVA
        Qd = self.bus["QD"].to_numpy(dtype=float) / self.baseMVA

        # Online Generators
        gen_on = self.gen[self.gen["GEN_STATUS"] > 0]

        gen_sum = (
            gen_on.groupby("GEN_BUS")[["PG","QG"]]
            .sum()
            .reindex(self.bus2index, fill_value=0.0)
        )

        Pg = gen_sum["PG"].to_numpy(dtype=float) / self.baseMVA
        Qg = gen_sum["QG"].to_numpy(dtype=float) / self.baseMVA

        # Net specified injection
        P_spec = Pg - Pd
        Q_spec = Qg - Qd

        return P_spec, Q_spec


    def make_ybus(self, ignore_shift=False):
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

        nb = len(self.bus)
        Ybus = np.zeros(
            (nb, nb),
            dtype=complex
        )

        # ----------------------
        # Branch Admittance
        # ----------------------

        for _,line in self.branch.iterrows():
            # Skip out-of-service branches
            if line["BR_STATUS"] == 0:
                continue

            # Bus Index in External World (Real Bus ID)
            from_bus = int(line["F_BUS"])
            to_bus = int(line["T_BUS"])

            # Bus Index in Python (0-based)
            i,j = self.bus2index[from_bus], self.bus2index[to_bus]

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
            if ignore_shift:
                shift = 0
            else:
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

        for _, row in self.bus.iterrows():

            bus_id = int(row["BUS_I"])
            i = self.bus2index[bus_id]

            # The value here is based on MW / MVAr, not p.u.
            gs = row["GS"]
            bs = row["BS"]

            Ybus[i, i] += (
                gs + 1j * bs
            ) / self.baseMVA

        # print(Ybus)
        return Ybus






    