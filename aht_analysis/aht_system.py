# %%
from . import utils
import numpy as np
from typing import List
import json
import matplotlib.pyplot as plt
X = utils.PAULIS["X"]
Y = utils.PAULIS["Y"]
Z = utils.PAULIS["Z"]
I = utils.PAULIS["I"]

def get_seq_unitaries(sequence: List[str]):
    unitaries = []
    for composite in sequence:
        labels = utils.parse_composite_pulse(composite)
        u = np.eye(2)
        for label in labels:
            u = utils.label_to_unitary(label) @ u
        unitaries.append(u)
    return unitaries



# %%
def aht_disorder(
    sequence: List[str],
    num_cycle: int = 1,
) -> List[float]:
    unitaries = get_seq_unitaries(sequence)
    
    H_list = [Z for _ in range(len(sequence))]

    uik = I

    aht_scores = utils.toggle_cycles(uik, unitaries, H_list, num_cycle)

    return aht_scores


def aht_dipolar(
    sequence: List[str],
    num_cycle: int = 1,
) -> List[float]:
    unitaries = [np.kron(u, u) for u in get_seq_unitaries(sequence)]

    dipolar = np.kron(X, X) + np.kron(Y, Y) - np.kron(Z, Z)

    H_list = [dipolar for _ in range(len(unitaries))]

    uik = np.kron(I, I)

    aht_scores = utils.toggle_cycles(uik, unitaries, H_list, num_cycle)

    return aht_scores

# # %%
# import json
# with open('preset_sequences.json', 'r', encoding='utf-8') as file:
#     preset_sequences = json.load(file)['preset_sequences']

# seq_name = 'droid_r2d2'
# # seq_name = 'xy8'
# sequence = preset_sequences[seq_name]

# disorder_scores = aht_disorder(sequence, num_cycle=100)
# dipolar_scores = aht_dipolar(sequence, num_cycle=100)

# import matplotlib.pyplot as plt

# plt.plot(disorder_scores,linestyle='--', label = 'disorder')
# plt.plot(dipolar_scores, linestyle='-', label = 'dipolar')
# plt.legend()
# plt.show()
# # %%
# dipolar_scores

# # %%

# from load_seq import read_sorted_sequences
# with open('preset_sequences.json', 'r', encoding='utf-8') as file:
#     preset_sequences = json.load(file)['preset_sequences']

# seq_file = 'N2v1.csv'
# sorted_sequences = read_sorted_sequences(seq_file)
# n_seq = 11
# print(f'Sequence {n_seq}')
# seq_labels = sorted_sequences[n_seq]['seq_labels']

# disorder_scores = aht_disorder(seq_labels, num_cycle=100)
# dipolar_scores = aht_dipolar(seq_labels, num_cycle=100)


# plt.plot(disorder_scores, label='disorder')
# plt.plot(dipolar_scores, label='dipolar')
# plt.legend()
# plt.show()
# # %%
# unitaries = get_seq_unitaries(seq_labels)

# # %%
# u = np.eye(2)
# for i in range(len(unitaries)):
#     u = unitaries[i] @ u

# # %%
# print(u)
# %%

# %%
