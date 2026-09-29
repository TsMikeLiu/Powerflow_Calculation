# PyPowerFlow

A simple AC power flow solver implemented from scratch using **Python, NumPy, pandas, and SciPy**.

This project is designed for learning **power flow algorithms, numerical methods, and object-oriented programming (OOP)** in Python. Input data follows the **MATPOWER case format**.

## Implemented Methods

### Newton-Raphson Power Flow

- Full AC Newton-Raphson formulation
- Active and reactive power mismatch calculation
- Full Jacobian matrix construction
- Iterative voltage magnitude and angle updates

### P-Q Decoupled Power Flow

- Fixed \(B'\) and \(B''\) matrices
- Sequential \(P-\theta\) and \(Q-V\) corrections
- Reusable LU factorization using SciPy
- Constant coefficient matrices throughout the iteration

## Object-Oriented Structure

The project uses an object-oriented architecture to separate **power system data** from **power flow algorithms**.

### `PowerSystem`

The `PowerSystem` class stores and processes network data, including:

- MATPOWER-style `bus`, `gen`, and `branch` data
- External bus ID to internal index mapping
- Bus classification: Slack, PV, and PQ buses
- Bus admittance matrix \(Y_{\text{bus}}\)
- Specified active and reactive power injections

Example:

```python
system = PowerSystem(
    baseMVA,
    bus,
    gen,
    branch
)
```

System information can then be accessed through:

```python
system.Ybus
system.bus2index

system.slack
system.pv
system.pq

system.P_spec
system.Q_spec
```

### `PowerFlowSolver`

`PowerFlowSolver` is the base class for different power flow algorithms.

It provides common functionality such as:

- Voltage initialization
- Complex voltage construction
- Bus power calculation
- Power mismatch calculation
- Solver status and result storage

The current solver hierarchy is:

```text
PowerFlowSolver
│
├── NewtonRaphsonSolver
│
└── PQDecoupledSolver
```

Each solver maintains its own voltage state (`Vm`, `Va`) while sharing the same `PowerSystem` object.

Example:

```python
nr_solver = NewtonRaphsonSolver(system)
nr_solver.solve()

pq_solver = PQDecoupledSolver(system)
pq_solver.solve()
```

Results can be accessed directly from the solver object:

```python
solver.converged
solver.iteration

solver.V
solver.Vm
solver.Va

solver.P_calc
solver.Q_calc
```

## Project Structure

```text
PyPowerFlow/
├── cases/
│   └── case14.py
├── main.py
├── network.py
└── powerflow.py
```

- `cases/`: MATPOWER-style test cases
- `network.py`: `PowerSystem` class and network processing
- `powerflow.py`: Base solver and power flow algorithms
- `main.py`: Run and compare different power flow methods

## Data Flow

```text
MATPOWER Case
      │
      ▼
 pandas DataFrame
      │
      ▼
  PowerSystem
      │
      ├── Bus Mapping
      ├── Bus Classification
      ├── Ybus
      └── Specified Power
      │
      ▼
PowerFlowSolver
      │
      ├── Newton-Raphson
      │
      └── P-Q Decoupled
      │
      ▼
 Power Flow Results
```

The program distinguishes between:

- **External bus IDs**: MATPOWER `BUS_I`
- **Internal indices**: contiguous Python indices `0, 1, ..., N-1`

Therefore, bus IDs are not required to be consecutive or start from 1.

## Dependencies

```bash
pip install numpy pandas scipy
```

Main libraries:

- **NumPy** — numerical arrays and linear algebra
- **pandas** — MATPOWER data processing
- **SciPy** — LU factorization and repeated linear system solutions

## Usage

Run:

```bash
python main.py
```

Example:

```python
system = PowerSystem(
    baseMVA,
    bus,
    gen,
    branch
)

solver = NewtonRaphsonSolver(
    system,
    tol=1e-8,
    max_iter=20,
    flat_start=True
)

solver.solve()
```

The solver provides:

- Bus voltage magnitude
- Bus voltage angle
- Active power injection
- Reactive power injection
- Convergence status
- Number of iterations

## Current Test Case

- IEEE 14-Bus System

The input data follows the MATPOWER case format.

## Notes

The current P-Q Decoupled implementation uses a simplified fixed-\(B\) formulation for educational purposes.

The \(B'\) and \(B''\) matrices currently:

- Ignore branch resistance
- Ignore line charging
- Ignore bus shunts
- Ignore transformer phase shifts
- Retain transformer tap ratios

This implementation should therefore be distinguished from the exact **MATPOWER FDXB/FDBX Fast Decoupled Power Flow** formulations.

## Planned

- Fast Decoupled Power Flow (FDXB/FDBX)
- Sparse matrix implementation
- Additional MATPOWER test cases
- Generator output calculation
- Reactive power limit handling
- Improved result and convergence reporting