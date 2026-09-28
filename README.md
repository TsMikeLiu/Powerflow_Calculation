# PyPowerFlow

A simple AC power flow solver implemented from scratch using **Python, NumPy, pandas, and SciPy**.

The project is designed for learning numerical methods and Python programming for power system analysis. Input data follows the **MATPOWER case format**.

## Implemented Methods

- Newton-Raphson Power Flow
- P-Q Decoupled Power Flow
  - Fixed \(B'\) and \(B''\) matrices
  - Sequential \(P-\theta\) and \(Q-V\) corrections
  - Reusable LU factorization

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
- `network.py`: Network processing and Ybus construction
- `powerflow.py`: Power flow algorithms
- `main.py`: Run and compare different power flow methods

## Dependencies

```bash
pip install numpy pandas scipy
```

## Usage

Run:

```bash
python main.py
```

The program outputs bus voltage magnitude, voltage angle, active power injection, and reactive power injection.

## Current Test Case

- IEEE 14-Bus System

## Planned

- Fast Decoupled Power Flow
- Sparse matrix implementation
- Additional MATPOWER test cases