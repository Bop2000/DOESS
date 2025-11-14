from typing import Dict, Any, Optional, List, Tuple
import os
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
import numpy as np

from doess import experiment

# Load preset sequences from JSON file
with open('preset_sequences.json', 'r') as file:
    PRESET_SEQUENCES = json.load(file)['preset_sequences']

def ensure_config_folder_exists(path: str) -> None:
    """Ensure the configuration folder exists."""
    os.makedirs(path, exist_ok=True)

def load_json_file(file: str, path: str):
    """Load data from a JSON file."""
    full_path = os.path.join(
        path, 
        f"{file}" if file.endswith('.json') else f"{file}.json"
    )
    
    with open(full_path, 'r') as f:
        full_data = json.load(f)
    
    simulation_data = full_data.get('simulation_data', {})
    plotting_params = full_data['all_params'].get('plotting', {})
    return full_data, simulation_data, plotting_params

class DataAnalyzer:
    def __init__(
        self,
        file: str = 'config1',
        path: str = 'configurations',
        exp = experiment.Experiment
    ):
        """
        Initialize a DataAnalyzer instance.

        Args:
            file (str): Name of the JSON file (with or without .json extension).
            path (str): Path of the folder containing the JSON file.
        """
        self.full_path = os.path.join(
            path, 
            f"{file}" if file.endswith('.json') else f"{file}.json"
        )
        self.full_data, self.simulation_data, self.plotting_params\
             = load_json_file(file, path)

        self.new_seq_name = None
        self.sequence_labels = None

        self.exp = exp(file, path)

        
    def run_sequence(self, input_sequence, repetitions):
        """
        Add the decay curve data to the config.json file and save it.

        Args:
            seq_name (str): Name of the sequence.
            seq_labels (List[str]): Labels for the sequence.
            results (Dict[str, Dict[str, List[float]]]): Results data.
            force_overwrite (bool): If True, overwrite existing sequence data. Default is False.

        Raises:
            ValueError: If sequence name already exists and force_overwrite is False.
        """
        self.exp.set_sequence(input = input_sequence)

        if repetitions == 0:
            return self.exp.sequence_labels, None
        else:
            try:    
                results = self.exp.run(
                    repetitions = repetitions,
                    min_N = self.plotting_params['min_N'],
                    step_N = self.plotting_params['step_N'],
                    num_points= self.plotting_params['num_points']
                )
            except KeyError:
                results = self.exp.run(
                    repetitions = repetitions,
                    max_time = self.plotting_params['max_time'],
                    min_time = self.plotting_params['min_time'],
                    num_points= self.plotting_params['num_points']
                )
                
            return self.exp.sequence_labels, results


    def save_data(
        self, 
        seq_name,
        seq_labels,
        simulation_results,
        force_overwrite: bool = False
    ) -> None:
        """Save current data to the JSON file."""
        if seq_name in self.simulation_data and not force_overwrite:
            raise ValueError(
                f"Sequence '{self.new_seq_name}' already exists."
                "Use force_overwrite=True to overwrite.")
        else:
            self.simulation_data[seq_name] ={
                'pulse_labels': seq_labels,
                'results': simulation_results
            }
    
        self.full_data['simulation_data'] = self.simulation_data

        ensure_config_folder_exists(os.path.dirname(self.full_path))
        with open(self.full_path, 'w') as file:
            json.dump(self.full_data, file, indent=4)
        print(
            f"Simulation data for sequence '{seq_name}'"
            f"saved to {self.full_path}"
        )


    def plot(
        self,
        seq_labels,
        results,
        max_time=None,
        baselines=['droid48', 'droid24', 'xy8'],
        axis='avg_over_axes',
        input_label = None,
        baseline_only = False,
    ):  
        # Set the style and color palette using Seaborn
        sns.set_style("whitegrid")
        colors = sns.color_palette(["black", "red", "#006400", "blue", "orange", "purple", "teal", "magenta"])  # Manually selected distinct colors
        
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))

        if not baseline_only:
            if max_time is not None:
                x_data = [i for i in results['timesteps'] if i <= max_time]
            else:
                max_time = max(results['timesteps'])
                x_data = results['timesteps']
            # Plot the new sequence

            y_data = results[axis]['mean'][:len(x_data)]
            upper_ci = results[axis]['upper_ci'][:len(x_data)]
            lower_ci = results[axis]['lower_ci'][:len(x_data)]

            if input_label is None:
                input_label = 'ML-seq'
            
            sns.lineplot(x=x_data, y=y_data, ax=ax, color=colors[0], label=input_label, marker='o', linewidth=3)
            ax.fill_between(x_data, lower_ci, upper_ci, alpha=0.1, color=colors[0])

        # Plot the baselines
        for i, baseline_seq_name in enumerate(baselines, 1):
            x_data = self.simulation_data[baseline_seq_name]['results']['timesteps']
            if max_time is not None:
                x_data = [i for i in x_data if i <= max_time]
            else:
                max_time = x_data[-1]
            y_data = self.simulation_data[baseline_seq_name]['results'][axis]['mean'][:len(x_data)]

            linestyle = '-' if i != 1 else '-'
            linewidth = 3 if i == 1 else 2
            alpha = 1 if i == 1 else 0.7
            
            sns.lineplot(x=x_data, y=y_data, ax=ax, color=colors[i], label=baseline_seq_name, 
                        linestyle=linestyle,  marker='o', linewidth=linewidth, alpha=alpha)

        # Customize the plot
        if axis == 'avg_over_axes':
            saturation = 2/3
        elif axis == 'Z':
            saturation = 1
        elif axis in ['X', 'Y']:
            saturation = 0.5
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        ax.axhline(y=saturation, color="#0072B2", linestyle='--', linewidth=2, alpha=1)
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel('Mean', fontsize=24)
        ax.set_title(f'Comparison: {axis}', fontsize=28, fontweight='bold')
        ax.legend(fontsize=18, loc='best')
        ax.grid(True, which='both', linestyle='--', linewidth=0.6)

        # Set larger tick parameters
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Set xticks from 0 to max_time with steps that ensure a reasonable number of ticks
        max_ticks = 10  # Maximum number of ticks we want to see
        step = max(1, int(max_time / max_ticks))  # Ensure step is at least 1
        xticks = list(range(0, int(max_time) + 1, step))
        
        # If there are still too many ticks, increase the step size
        while len(xticks) > max_ticks:
            step *= 2
            xticks = list(range(0, int(max_time) + 1, step))
        
        ax.set_xticks(xticks)
        ax.set_ylim(0.45, 1.01)

        plt.tight_layout()
        plt.show()

        
        

