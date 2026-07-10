# well-stirred combustor, ground truth for the surrogates. tau -> X_CO.

import cantera as ct

MECH = "gri30.yaml"
T_IN = 300.0       # inlet (fresh reactant) temperature [K]
P = ct.one_atm     # pressure [Pa]
PHI = 1.0          # equivalence ratio (stoichiometric CH4/air)
FUEL = "CH4"
OXIDIZER = "O2:1, N2:3.76"


def run_reactor(tau: float) -> float:
    # returns steady-state X_CO for residence time tau [s]
    gas = ct.Solution(MECH)

    gas.TP = T_IN, P
    gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)
    inlet = ct.Reservoir(gas)

    gas.equilibrate("HP")  # start hot so it stays ignited
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
