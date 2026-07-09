"""Ground-truth simulator for this project: a well-stirred combustor, a
small chamber that fresh methane and air flow into continuously while hot
exhaust flows out, held at constant pressure. Chemistry is tracked with
GRI-Mech 3.0. This follows Cantera's standard combustor recipe: a hot
reactor fed fresh reactants at a controlled mass flow rate and drained to
an exhaust reservoir at constant pressure.

Input is residence time tau [s], how long a parcel of gas sits in the
chamber on average. Output is the CO mole fraction at steady state. The
shape isn't a simple curve that flattens out: at short tau the reactor is
near blowout and CO is low, at intermediate tau partial oxidation produces
a lot of CO, and at long tau that CO gets oxidized further into CO2 and
drops back down.
"""

import cantera as ct

MECH = "gri30.yaml"
T_IN = 300.0       # inlet (fresh reactant) temperature [K]
P = ct.one_atm     # pressure [Pa]
PHI = 1.0          # equivalence ratio (stoichiometric CH4/air)
FUEL = "CH4"
OXIDIZER = "O2:1, N2:3.76"


def run_reactor(tau: float) -> float:
    """Run the well-stirred combustor at residence time `tau` [s] and
    return the steady-state CO mole fraction. This is the real simulator
    that everything else in the project treats as ground truth.
    """
    gas = ct.Solution(MECH)

    # Fresh reactant mixture (used for the inlet reservoir)
    gas.TP = T_IN, P
    gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)
    inlet = ct.Reservoir(gas)

    # Reactor initially filled with hot equilibrium products so it stays
    # ignited (matches the standard Cantera combustor recipe).
    gas.equilibrate("HP")
    combustor = ct.IdealGasReactor(gas)
    combustor.volume = 1.0  # m^3; arbitrary, tau is set via mdot below

    exhaust = ct.Reservoir(gas)

    def mdot(t):
        return combustor.mass / tau

    inlet_mfc = ct.MassFlowController(inlet, combustor, mdot=mdot)
    ct.PressureController(combustor, exhaust, primary=inlet_mfc, K=0.01)

    sim = ct.ReactorNet([combustor])
    sim.rtol = 1e-9
    sim.atol = 1e-20
    sim.advance_to_steady_state()

    return combustor.thermo["CO"].X[0]


if __name__ == "__main__":
    for tau in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        print(f"tau={tau:>7.4f} s  ->  X_CO={run_reactor(tau):.6e}")
