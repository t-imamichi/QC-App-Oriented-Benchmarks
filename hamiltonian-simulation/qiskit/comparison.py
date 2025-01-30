import argparse
import json
import os

import numpy as np
from qiskit.quantum_info import Operator

from hamiltonian_simulation_kernel import HeisenbergHamiltonianKernel

np.random.seed(0)

# Import precalculated data to compare against
filename = os.path.join(
    os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json"
)
with open(filename, "r") as file:
    data = file.read()
precalculated_data = json.loads(data)


def run(
    num_qubits: int = 2,
    hamiltonian: str = "heisenberg",
    method: int = 1,
    use_XX_YY_ZZ_gates: bool = False,
    random_pauli_flag: bool = False,
    init_state: str | None = None,
):
    # check for valid Hamiltonian name
    if not (hamiltonian == "heisenberg" or hamiltonian == "tfim"):
        print(f"ERROR: invalid Hamiltonian name: {hamiltonian}")
        return

    # set the initial method if no initial state argument is given by user.
    if init_state is None:
        if hamiltonian == "tfim":
            init_state = "ghz"
        else:
            init_state = "checkerboard"

    # Set the flag to use an XX YY ZZ shim if given
    if use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")

    # since method 1 and 2 use pre-calculated data, cannot go above 12 qubits
    if method == 1 or method == 2:
        if num_qubits > 12:
            print("ERROR: cannot execute method 1 or 2 above 12 qubits")
            return

    #######################################################################
    # Parameters of simulation
    #### CANNOT BE MODIFIED W/O ALSO MODIFYING PRECALCULATED DATA #########
    w = precalculated_data["w"]  # Strength of disorder
    k = precalculated_data["k"]  # Trotter error.
    # A large Trotter order approximates the Hamiltonian evolution better.
    # But a large Trotter order also means the circuit is deeper.
    # For ideal or noise-less quantum circuits, k >> 1 gives perfect Hamiltonian simulation.
    t = precalculated_data["t"]  # Time of simulation

    # Precalculated random numbers between [-1, 1]
    hx = precalculated_data["hx"][:num_qubits]
    hz = precalculated_data["hz"][:num_qubits]
    #######################################################################

    qcs = []
    for circuit_type in range(3):
        if circuit_type == 0:
            use_XX_YY_ZZ_gates = True
            use_pauli_evolution = False
        elif circuit_type == 1:
            use_XX_YY_ZZ_gates = False
            use_pauli_evolution = False
        else:
            use_XX_YY_ZZ_gates = False
            use_pauli_evolution = True
        qc_object = HeisenbergHamiltonianKernel(
            num_qubits,
            K=k,
            t=t,
            hamiltonian=hamiltonian,
            w=w,
            hx=hx,
            hz=hz,
            use_XX_YY_ZZ_gates=use_XX_YY_ZZ_gates,
            use_pauli_evolution=use_pauli_evolution,
            method=method,
            random_pauli_flag=random_pauli_flag,
            init_state=init_state,
        )
        qcs.append(qc_object.overall_circuit())
    qcs[0].metadata["name"] = "Naive circuit"
    qcs[1].metadata["name"] = "Optimized circuit"
    qcs[2].metadata["name"] = "Circuit with PauliEvolutionGate"

    for qc in qcs:
        print(qc.metadata["name"])
        print(qc)
        qc.remove_final_measurements()

    for i in range(3):
        for j in range(i + 1, 3):
            if Operator(qcs[i]).equiv(qcs[j]):
                print(
                    f"OK: {qcs[i].metadata['name']} and {qcs[j].metadata['name']} are equivalent"
                )
            else:
                print(
                    f"NG: {qcs[i].metadata['name']} and {qcs[j].metadata['name']} are not equivalent"
                )


def get_args():
    parser = argparse.ArgumentParser(
        description="Comparison of Hamiltonian simulation circuits"
    )
    parser.add_argument(
        "--num-qubits", "-n", default=3, help="Number of qubits", type=int
    )
    parser.add_argument(
        "--hamiltonian",
        "--ham",
        default="heisenberg",
        help="Name of Hamiltonian",
        type=str,
    )
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument(
        "--random_pauli_flag", "-ranp", action="store_true", help="random pauli flag"
    )
    parser.add_argument("--init_state", "-init", default=None, help="initial state")
    return parser.parse_args()


# if main, execute method
if __name__ == "__main__":
    args = get_args()

    run(
        num_qubits=args.num_qubits,
        hamiltonian=args.hamiltonian,
        method=args.method,
        random_pauli_flag=args.random_pauli_flag,
        init_state=args.init_state,
    )
