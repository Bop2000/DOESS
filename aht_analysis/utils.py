import numpy as np
from typing import Tuple, Union, List
import re
from functools import reduce
import random
from scipy.linalg import expm
from numpy.linalg import norm
import matplotlib.pyplot as plt
import json

PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "I": np.eye(2, dtype=np.complex128)
}


def parse_composite_pulse(composite):
    return [pulse for pulse in composite.split(";")]

def flatten_seq_labels(seq_labels: List[str]) -> List[str]:
    labels = []
    for composite in seq_labels:
        labels.extend(parse_composite_pulse(composite))

    return labels

def remove_null_pulses(seq_labels: List[str]) -> List[str]:
    return [label for label in seq_labels if label != "null"]
    
# parse_composite_pulse("+X,pi/2;-Y,pi/2") # ['+X,pi/2', '-Y,pi/2']
# # parse_composite_pulse("+X,pi/2") # ['+X,pi/2']


def parse_pulse_string(pulse_str: str) -> Tuple[float, float]:
        if pulse_str == "null":
            return ("X", 0)  # Special case for null pulse
        
        direction, angle_str = pulse_str.split(',')
        
        weight = +1 if direction[0] == '+' else -1
        axis = direction[1].upper()
   
        match = re.match(r'(\d+)?pi(/(\d+))?', angle_str)
        if match:
            numerator = int(match.group(1)) if match.group(1) else 1
            denominator = int(match.group(3)) if match.group(3) else 1
            weight = numerator / denominator * weight
        else:
            raise ValueError(f"Unable to parse angle: {angle_str}")
        
        return axis, weight
# parse_pulse_string("-Y,pi/2") # ('Y', -0.5)
# parse_pulse_string("+X,pi/3") # ('X', 0.3333333333333333)


def clean_near_zero(x, tol=1e-14):
    x.real[np.abs(x.real) < tol] = 0.0
    x.imag[np.abs(x.imag) < tol] = 0.0
    return x

def label_to_unitary(label, tol=1e-16):
    axis, weight = parse_pulse_string(label)

    unitary = (
        np.cos(weight * np.pi / 2) * PAULIS["I"] 
        - 1j * np.sin(weight * np.pi / 2) * PAULIS[axis]
    )
    
    return clean_near_zero(unitary)

# label_to_unitary("-X,pi/3") # array([[0.8660254+0.j , 0.       +0.5j],
#                           # [0.       +0.5j, 0.8660254+0.j ]])


def to_toggle(h, u):
    return u.conj().T @ h @ u

def one_toggle_cycle(uik, unitaries, H_list):
    avg_H = to_toggle(H_list[0], uik)
    for k, pulse_u in enumerate(unitaries[:-1]):
        uik = pulse_u @ uik
        avg_H += to_toggle(H_list[k+1], uik)
    avg_H /= len(unitaries)
    # last pulse does not affect the average Hamiltonian in the current cycle
    # but we need to update uik for the next cycle
    uik = unitaries[-1] @ uik
    return uik, avg_H

def toggle_cycles(uik, unitaries, H_list, num_cycle):
    avg_H_list = []
    for k in range(num_cycle):
        # updating uik in an accumulation manner
        uik, avg_H = one_toggle_cycle(uik, unitaries, H_list)
        # store the average Hamiltonian for each repetition of the sequence
        avg_H_list.append(avg_H)
    
    # calculate the frobenius norm as the for each repetition of the sequence
    # repeating sequence is regarded as a longer single sequence
    aht_scores = [
        norm(
            sum(avg_H_list[:i]) / i
        ) 
        for i in range(1, len(avg_H_list))
    ]

    return aht_scores


def norm(H: np.ndarray, tol: float = 1e-12) -> float:

    norm = min(
        np.linalg.norm(H, 'fro'),
        np.linalg.norm(H - np.eye(H.shape[0]), 'fro')
    )

    if H.shape[0] == 4:  # For two-qubit case
        X, Y, Z = PAULIS["X"], PAULIS["Y"], PAULIS["Z"]
        target = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)

        inner_product = np.trace(np.conjugate(target.T) @ H)
        target_norm = np.trace(np.conjugate(target.T) @ target)
        
        alpha = inner_product / target_norm
        
        # Calculate norm of difference to the reference matrix
        # alpha * (XX + YY + ZZ)
        diff_norm = np.linalg.norm(H - alpha * target, 'fro')

        norm = min(diff_norm, norm)

    return  norm

