# %%
from . import utils
import numpy as np
from typing import List

X = utils.PAULIS["X"]
Y = utils.PAULIS["Y"]
Z = utils.PAULIS["Z"]
I = utils.PAULIS["I"]


def get_pulse_hamiltonian(label: str) -> np.ndarray:
    axis, weight = utils.parse_pulse_string(label)
    return weight * utils.PAULIS[axis]

# %%
def aht_rotation(
    sequence: List[str],
    num_cycle: int = 1,
) -> List[float]:
    labels = utils.flatten_seq_labels(
        utils.remove_null_pulses(sequence)
    )

    unitaries = [utils.label_to_unitary(label) for label in labels]

    H_list = [get_pulse_hamiltonian(label) for label in labels]

    uik = I

    aht_scores = utils.toggle_cycles(uik, unitaries, H_list, num_cycle)
    
    return aht_scores

# %%
# import json
# with open('preset_sequences.json', 'r', encoding='utf-8') as file:
#     preset_sequences = json.load(file)['preset_sequences']

# seq_name = 'droid_r2d2'
# seq_name = 'droid48'
# sequence = preset_sequences[seq_name]

# aht_scores = aht_rotation(sequence, num_cycle=100)

# import matplotlib.pyplot as plt
# plt.ylim(0, 0.3)
# plt.plot(aht_scores)
# plt.show()

# %%
