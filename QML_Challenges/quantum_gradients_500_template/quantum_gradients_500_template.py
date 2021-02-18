#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    # QHACK #
    def get_state(params):
        """ Get the state before a measurement """
        qnode(params)
        return dev.state

    # Calculate the unshifted state (its conjugate transpose)
    state_unshifted = np.conjugate(get_state(params)).T

    def shift_vector(i):
        vector = np.zeros(6)
        vector[i] = 1
        return vector

    metric_tensor = np.zeros((6, 6))
    
    for i in range(6):
        for j in range(i + 1):
            
            state_shifted_1 = get_state(params + (shift_vector(i) + shift_vector(j)) * np.pi/2)
            state_shifted_2 = get_state(params + (shift_vector(i) - shift_vector(j)) * np.pi/2)
            state_shifted_3 = get_state(params + (-shift_vector(i) + shift_vector(j)) * np.pi/2)
            state_shifted_4 = get_state(params - (shift_vector(i) + shift_vector(j)) * np.pi/2)

            metric_tensor[i, j] = (
                - np.abs(np.dot(state_unshifted, state_shifted_1))**2
                + np.abs(np.dot(state_unshifted, state_shifted_2))**2
                + np.abs(np.dot(state_unshifted, state_shifted_3))**2
                - np.abs(np.dot(state_unshifted, state_shifted_4))**2
            ) / 8

            if i != j:
                metric_tensor[j, i] = metric_tensor[i, j]

    grad = qml.grad(qnode)
    gradient = grad(params)[0]

    metric_tensor_inv = np.linalg.inv(metric_tensor)

    natural_grad = np.dot(metric_tensor_inv, gradient)

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
