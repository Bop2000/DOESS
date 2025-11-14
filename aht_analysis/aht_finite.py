# %%
from . import utils
import numpy as np
from typing import List
from scipy.linalg import expm

X = utils.PAULIS["X"]
Y = utils.PAULIS["Y"]
Z = utils.PAULIS["Z"]
I = utils.PAULIS["I"]


def get_finite_pulse_hamiltonian(
    H: np.ndarray,
    label: str,
    num_points: int = 2000,
) -> np.ndarray:
    axis, weight = utils.parse_pulse_string(label)

    sigma_nu = utils.PAULIS[axis] * weight / abs(weight)
    total_angle = np.pi * abs(weight)
    
    avg_H = np.zeros_like(H)
    for i in range(num_points):
        theta = i * total_angle / num_points
        u_theta = np.cos(theta/2) * I - 1j * np.sin(theta/2) * sigma_nu

        if H.shape[0] == 4: # two-qubit case
            u_theta = np.kron(u_theta, u_theta)
            
        avg_H += u_theta.conj().T @ H @ u_theta

    return avg_H / num_points

def aht_finite(
    sequence: List[str],
    H: np.ndarray = Z,
    num_cycle: int = 1,
) -> List[float]:
    is_two_qubit = (H.shape[0] == 4)

    labels = utils.flatten_seq_labels(
        utils.remove_null_pulses(sequence)
    )
    unitaries = [utils.label_to_unitary(label) for label in labels]
    if is_two_qubit:
        unitaries = [np.kron(u, u) for u in unitaries]
    
    H_list = [get_finite_pulse_hamiltonian(H, label) for label in labels]

    uik = np.kron(I, I) if is_two_qubit else I

    aht_scores = utils.toggle_cycles(uik, unitaries, H_list, num_cycle)
    
    return aht_scores

# %%
# import json
# with open('preset_sequences.json', 'r', encoding='utf-8') as file:
#     preset_sequences = json.load(file)['preset_sequences']

# seq_name = 'droid_r2d2'
# # seq_name = 'droid48'
# # seq_name = 'xy8'
# sequence = preset_sequences[seq_name]

# H = Z
# aht_scores_disorder = aht_finite(sequence, H, num_cycle=100)

# H = np.kron(X, X) + np.kron(Y, Y) - np.kron(Z, Z)
# aht_scores_dipolar = aht_finite(sequence, H, num_cycle=100)

# import matplotlib.pyplot as plt
# plt.title(f'{seq_name}')

# plt.plot(aht_scores_disorder, label='disorder')
# plt.plot(aht_scores_dipolar, label='dipolar')
# plt.legend()
# plt.show()
# # %%
# aht_scores_dipolar
# %%
