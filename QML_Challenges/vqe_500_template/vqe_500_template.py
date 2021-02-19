#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def variational_ansatz(params, wires):
    """
    Args:
        params (np.ndarray): An array of floating-point numbers with size (n, 3),
            where n is the number of parameter sets required (this is determined by
            the problem Hamiltonian).
        wires (qml.Wires): The device wires this circuit will run on.
    """
    n_qubits = len(wires)
    n_rotations = len(params)

    if n_rotations > 1:
        n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits

        # Alternating layers of unitary rotations on every qubit followed by a
        # ring cascade of CNOTs.
        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

        # There may be "extra" parameter sets required for which it's not necessarily
        # to perform another full alternating cycle. Apply these to the qubits as needed.
        extra_params = params[-n_extra_rots:, :]
        extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
        qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
    else:
        # For 1-qubit case, just a single rotation to the qubit
        qml.Rot(*params[0], wires=wires[0])

def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #
    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))

    # Initialize the quantum device
    dev = qml.device("default.qubit", wires=num_qubits)

    # Set up a cost function
    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)

    # Set up an optimizer
    #opt = qml.QNGOptimizer(0.01, diag_approx=False, lam=0.001)
    #opt = qml.GradientDescentOptimizer(0.8)
    opt = qml.QNGOptimizer(0.1, diag_approx=False, lam=0.001)

    # Run the VQE by iterating over many steps of the optimizer
    max_iterations = 30
    conv_tol = 1e-09

    energies = []

    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        energies.append(cost_fn(params))
        conv = np.abs(energies[-1] - prev_energy)

        if conv <= conv_tol:
            break

    energy = energies[-1]

    # QHACK #

    # Return the ground state energy
    return energies, dev.state


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)
    
    all_energies = []

    # QHACK #
    
    # First run
    ground_state_energies_1, final_state_1 = run_vqe(H)
    
    all_energies.append(ground_state_energies_1)
    
    energies[0] = ground_state_energies_1[-1]
    
    # 2nd round
    # Create the projection operator
    projection_operator = np.kron(np.conjugate(final_state_1[:, np.newaxis]), final_state_1)
    
    # new Hamiltonian: H_1 = H - E_g P
    coeffs, observables = H.terms
    ground_state_energy = ground_state_energies_1[-1]
    
    pcoeffs, pobs = qml.utils.decompose_hamiltonian(projection_operator)
    new_coeffs = coeffs + [-coeff * ground_state_energy for coeff in pcoeffs]
    new_observables = observables + pobs

    new_H = qml.Hamiltonian(coeffs=new_coeffs, observables=new_observables)

    ground_state_energies_2, final_state_2 = run_vqe(new_H)
    all_energies.append(ground_state_energies_2)
    
    energies[1] = ground_state_energies_2[-1]
    
    # 3rd round
    # Create the projection operator
    projection_operator = np.kron(np.conjugate(final_state_2[:, np.newaxis]), final_state_2)
    
    # new Hamiltonian: H_1 = H - E_g P
    
    coeffs, observables = new_H.terms
    ground_state_energy = ground_state_energies_2[-1]
    
    pcoeffs, pobs = qml.utils.decompose_hamiltonian(projection_operator)
    new_coeffs = coeffs + [-coeff * ground_state_energy for coeff in pcoeffs]
    new_observables = observables + pobs

    new_H = qml.Hamiltonian(coeffs=new_coeffs, observables=new_observables)

    ground_state_energies_3, final_state_3 = run_vqe(new_H)
    all_energies.append(ground_state_energies_3)
    
    energies[2] = ground_state_energies_3[-1]
    
    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
