import json
import os
import numpy as np
from typing import Dict, Any
from datetime import datetime
from collections import OrderedDict


from . import awg



def get_spin_params() -> Dict[str, Any]:
    # disorder and detuning are frequencies in MHz
    return {
        'num_spins': 5,
        'density_ppm': 0.1,
        'disorder': 0.5,
        'interaction_type': {
            'XX': 1.0,
            'YY': 1.0,
            'ZZ': -1.0
        },
        'J0': 2 * np.pi * 52e-27,
        'detuning': 0
    }

def get_pulse_params() -> Dict[str, float]:
    return {
        'duration_error': 20 / 100, # ratio
        'phase_error': 0.02, # *(np.pi)
        "rabi_freq": 10 # MHz,
    }

def get_basic_pulse_labels() -> Dict[int, str]:
    return {
        0: 'null', 1: '+X,pi', 2: '+Y,pi', 3: '-X,pi', 4: '-Y,pi', 
        5: '+X,pi/2', 6: '+Y,pi/2', 7: '-X,pi/2', 8: '-Y,pi/2'
    }

def get_training_params() -> Dict[str, Any]:
    """Parameters for calculating performance metrics in objective.py"""
    return {
        'max_time': 2000,
        'min_time': 0,
        'num_points': 11,
        'critical_points': [0.8, 2/3],
        'fit_points': 1000
    }

def get_plotting_params() -> Dict[str, Any]:
    return {
        'max_time': 2000,
        'min_time': 0,
        'num_points': 201
    }


def get_all_candidates():
    basic_pulse_labels = get_basic_pulse_labels()

    # generate all possible unique composite pulses
    unitaries = [
        awg.Pulse(rabi=np.pi, pulse_str=name).unitary() 
        for key, name in basic_pulse_labels.items()
    ]

    candidates = {}
    for index, pulse_str in basic_pulse_labels.items():
        candidates[index] = [index]
    
    def new_unitary(u1, unitaries) -> bool:
        for u2 in unitaries:
            if np.allclose(u1, u2) or np.allclose(u1, -u2):
                return False
        return True

    for i in range(1, len(basic_pulse_labels)):
        for j in range(1, len(basic_pulse_labels)):
            u = np.dot(unitaries[j], unitaries[i])
            if new_unitary(u, unitaries):
                unitaries.append(u)
                candidates[len(unitaries)-1] = [i, j]
            else:
                continue

    return candidates

def ensure_config_folder_exists(folder_name: str = "configurations") -> None:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created '{folder_name}' folder.")

def get_valid_file_name(folder_name: str) -> str:
    while True:
        file_name = input("Please enter the name for the JSON file (without extension): ").strip() + ".json"
        full_path = os.path.join(folder_name, file_name)
        if os.path.exists(full_path):
            overwrite = input(
                f"The file '{full_path}' already exists. "
                "Do you want to overwrite it? (yes/no): "
            ).strip().lower()
            if overwrite == 'yes':
                return full_path
            print("Please choose another name.")
        else:
            return full_path

def save_params_to_json(file_path: str, data: OrderedDict) -> None:
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data has been saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError as e:
        print(f"An error occurred while encoding JSON: {e}")


def main() -> None:
    config_folder = "configurations"
    ensure_config_folder_exists(config_folder)
    
    file_path = get_valid_file_name(config_folder)
    
    all_params = OrderedDict([
        ('spin', get_spin_params()),
        ('pulse', get_pulse_params()),
        ('training', get_training_params()),
        ('plotting', get_plotting_params())
    ])

    full_data = OrderedDict([
        ('timestamp', datetime.now().isoformat()),
        ('all_params', all_params),
        ('labels', get_basic_pulse_labels()),
        ('candidates', get_all_candidates()),
        ('simulation_data', {})
    ])

    save_params_to_json(file_path, full_data)

if __name__ == "__main__":
    main()