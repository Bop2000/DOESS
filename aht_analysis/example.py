# %%
import json
import matplotlib.pyplot as plt
import numpy as np
# %%
from aht_system import aht_disorder
from aht_system import aht_dipolar
from aht_rotation import aht_rotation
from aht_finite import aht_finite

from load_seq import read_sorted_sequences

import utils

X = utils.PAULIS['X']
Y = utils.PAULIS['Y']
Z = utils.PAULIS['Z']
I = utils.PAULIS['I']


def get_metric_data(seq_name, seq_labels, num_cycle=200):
    disorder = aht_disorder(seq_labels, num_cycle=num_cycle)
    dipolar = aht_dipolar(seq_labels, num_cycle=num_cycle)
    rotation = aht_rotation(seq_labels, num_cycle=num_cycle)

    H = Z
    finite_disorder = aht_finite(seq_labels, H, num_cycle=num_cycle)

    H = np.kron(X, X) + np.kron(Y, Y) - np.kron(Z, Z)
    finite_dipolar = aht_finite(seq_labels, H, num_cycle=num_cycle)


    return {
        'name': seq_name,
        'disorder': disorder,
        'dipolar': dipolar,
        'rotation': rotation,
        'finite_disorder': finite_disorder,
        'finite_dipolar': finite_dipolar
    }


# %%
with open('preset_sequences.json', 'r', encoding='utf-8') as file:
    preset_sequences = json.load(file)['preset_sequences']

seq_file = 'N2v1.csv'
sorted_sequences = read_sorted_sequences(seq_file)
# %%
from visualization import plot_sequence_metrics

r2d2_metrics = get_metric_data('r2d2', preset_sequences['droid_r2d2'])

# droid48_metrics = get_metric_data('droid48', preset_sequences['droid48'])

# droid24_metrics = get_metric_data('droid24', preset_sequences['droid24'])
# xy8_metrics = get_metric_data('xy8', preset_sequences['xy8'])

# plot_sequence_metrics(droid24_metrics)
# plot_sequence_metrics(droid48_metrics)
# plot_sequence_metrics(xy8_metrics)
# %%
for n_seq in range(len(sorted_sequences)):
    seq_name = f"# {n_seq} with score: {sorted_sequences[n_seq]['score']:.4f}"
    seq_labels = sorted_sequences[n_seq]['seq_labels']
    metrics = get_metric_data(seq_name, seq_labels)
    plot_sequence_metrics(
        metrics, 
        benchmarks=[r2d2_metrics], 
        path=f"figures/sequence_{n_seq}.png"
    )
    break

# %%
# plot_sequence_metrics(metrics, benchmark_metrics)

# # %%
# metrics
# # %%
# r2d2_metrics['finite_dipolar'][:5]


# # %%
# n_seq = 123
# metrics = get_metric_data(f"# {n_seq}", sorted_sequences[n_seq]['seq_labels'])
# metrics['dipolar'][:5]
# # %%
# sorted_sequences[n_seq]['compensations']
# # %%
