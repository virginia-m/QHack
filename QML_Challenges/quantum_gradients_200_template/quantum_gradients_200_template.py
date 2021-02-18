#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    shift_value = np.pi/4

    def parameter_shift_term(circuit, weights, i):
        shifted = weights.copy()
        shifted[i] += shift_value * 2
        forward = circuit(shifted)  # forward evaluation

        shifted[i] -= 2 * shift_value * 2
        backward = circuit(shifted) # backward evaluation

        return 0.5 * (forward - backward) / np.sin(shift_value*2), forward, backward
    
    forwards = np.zeros(5)
    backwards = np.zeros(5)
    for i in range(5):
        gradient[i], forwards[i], backwards[i] = parameter_shift_term(circuit, weights, i)
    
    def shift_vector(i):
        vector = np.zeros(5)
        vector[i] = 1
        return vector
    
    circuit_unshifted = circuit(weights)
    
    def evaluate_circuit(shifts):
        if np.any(shifts != 0) and np.all(shifts != 2*np.pi):
            return circuit(weights + shifts)
        return circuit_unshifted
  
    for i in range(5):
        for j in range(i+1):
            i_shift = shift_value * shift_vector(i)
            j_shift = shift_value * shift_vector(j)

            if i == j:
                hessian[i, i] = 0.25 * (
                    forwards[i] + backwards[i] - 2 * circuit_unshifted
                ) / np.sin(shift_value)**2

            else:
                hessian[i, j] = 0.25 * (
                    evaluate_circuit(i_shift + j_shift)
                    - evaluate_circuit(i_shift - j_shift)
                    - evaluate_circuit(-i_shift + j_shift)
                    + evaluate_circuit(-i_shift - j_shift)
                ) / np.sin(shift_value)**2
            
                hessian[j, i] = hessian[i, j]
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
