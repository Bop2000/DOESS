from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.linalg import expm
from scipy import stats
from functools import reduce
import json
import os
import re
from fractions import Fraction

from .sequence import create_composite_pulse_label, divide_pulse_label
from . import sequence
from .objectives import get_all_objective_functions
from . import register

# Constants
AXES = ['X', 'Y', 'Z']

class simulate:
    def __init__(
        self, 
        filename: str = 'config1', 
        file_path: str = 'configurations'
    ):
        """
        Initialize an simulate instance.

        Args:
            filename (str): 
                Name of the configuration file (without .json extension).
            file_path (str): 
                Path to the configuration file directory.
        """
        self.filename = filename
        self.file_path = file_path
        self.all_params, self.all_basic_labels, self.candidates \
            = self._load_data()
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize Sequence and Ensemble components."""
        self.seq = sequence.Sequence(
            self.all_params, self.all_basic_labels, self.candidates
        )
        self.ens = register.Ensemble(self.all_params['spin'])
    
    def change_params(
        self, 
        new_spin_params: Optional[Dict[str, Any]] = None, 
        new_pulse_params: Optional[Dict[str, Any]] = None,
        new_training_params: Optional[Dict[str, Any]] = None,
        print_info: bool = True
    ) -> None:
        """
        Update simulate parameters and reinitialize components.

        Args:
            new_spin_params (Optional[Dict[str, Any]]): New spin configuration
                parameters.
            new_pulse_params (Optional[Dict[str, Any]]): New pulse
                configuration parameters.
            new_training_params (Optional[Dict[str, Any]]): New training
            print_info (bool): Whether to print updated simulate info after
                changes. Defaults to True.

        Raises:
            ValueError: If no parameters are provided for update.
        """
        if not any([new_spin_params, new_pulse_params, new_training_params]):
            raise ValueError(
                "At least one set of parameters must be provided for update."
            )

        new_params = {}
        if new_spin_params:
            new_params['spin'] = new_spin_params
        if new_pulse_params:
            new_params['pulse'] = new_pulse_params
        if new_training_params:
            new_params['training'] = new_training_params
        

        # Deep update of nested dictionaries
        self._deep_update(self.all_params, new_params)

        self._initialize_components()

        if print_info:
            self.print_info()

    @property
    def sequence_labels(self):
        return self.seq.seq_labels

    @property
    def aht_score(self):
        return self.seq.calculate_aht_score()
    
    @property
    def aht_s1_s2_s3(self):
        return self.seq.calculate_aht_s1_s2_s3()
    
    @property
    def aht_s4_s5(self):
        return self.seq.calculate_aht_s4_s5()

    def sample_hamiltonian_and_pulse_error(self) -> Tuple[
        np.ndarray, Dict[str, Dict[str, np.ndarray]]
    ]:
        """
        Sample the Hamiltonian and pulse errors.

        Returns:
            Tuple containing sampled Hamiltonian and applied unitaries.
        """
        h_free = self.ens.sample_hamiltonian()
        self.ens.hamiltonian = h_free

        all_used_noisy_unitaries = \
            self.seq.sample_unitaries_with_pulse_errors(ensemble = self.ens)

        return h_free.full(), all_used_noisy_unitaries

    def pre_check(
        self,
        repetitions: int = 50,
        confidence_level: float = 0.98
    ):
        training_params = self.seq.params['training']

        check_time = (
            training_params['max_time'] + training_params['min_time']
        ) / 2
        num_points = 1

        results = self.run(
            repetitions, check_time, check_time, num_points
        )
        mean = results['avg_over_axes']['mean'][0]
        lower_ci = results['avg_over_axes']['lower_ci'][0]
        upper_ci = results['avg_over_axes']['upper_ci'][0]
        return lower_ci, mean, upper_ci


    def _calculate_avg_fidelity(self, U: np.ndarray, axis: str) -> float:
        """
        Calculate the average fidelity for a given unitary and axis.

        Args:
            U: The unitary matrix.
            axis: The axis ('X', 'Y', or 'Z').

        Returns:
            The calculated average fidelity.
        """
        
        final_state = U @ self.ens.pauli_eigenstates[axis].full()

        operator = self.ens.global_paulis[axis].full()

        obs_expect = np.real(
            final_state.conj().T @ operator @ final_state
        )[0, 0] / self.ens.num_spins

        return (1 + obs_expect) / 2
    
    def _deep_update(
            self, original: Dict[str, Any], update: Dict[str, Any]
        ) -> None:
        """
        Recursively update a nested dictionary.

        Args:
            original (Dict[str, Any]): The original dictionary to update.
            update (Dict[str, Any]): The dictionary containing updates.
        """
        for key, value in update.items():
            if (isinstance(value, dict) and key in original 
                    and isinstance(original[key], dict)):
                self._deep_update(original[key], value)
            else:
                original[key] = value

 
    def _load_data(self):
        """Load data from the JSON file."""
        full_path = os.path.join(
            self.file_path, 
            f"{self.filename}.json"
        )
        try:
            with open(full_path, 'r') as file:
                full_data = json.load(file)
            params = full_data.get('all_params', {})
            labels = full_data.get('labels', {})
            candidates = full_data.get('candidates', {})

            labels = {int(key): value for key, value in labels.items()}
            candidates = {int(key): value for key, value in candidates.items()}
            return params, labels, candidates
        except FileNotFoundError:
            print(f"File {full_path} not found.")
            return None, None, None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {full_path}: {e}")
            return None, None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None, None
    
    def print_info(self):
        with open('preset_sequences.json', 'r') as file:
            PRESET_SEQUENCES = json.load(file)['preset_sequences']
            

        """Print all simulate information."""
        print('\n', '-'*50)
        print('all parameters:')
        for key, params in self.all_params.items():
            print(f"{key}: {params}")
            print('-'*50)

        print('\n', '-'*50)
        print('Selected building blocks for training ML models:')
        print('All basic pulse labels:\n', self.all_basic_labels)
        print('-'*50)
        # print('All (composite) candidates:\n', self.candidates)
        # print('-'*50)
        # print('-'*50, '\n')

        print('-'*50)
        print("Available preset sequences:")
        print(list(PRESET_SEQUENCES.keys()))
        print('-'*50, '\n')



class simulator(simulate):
    def __init__(
        self, 
        filename: str = 'config1', 
        file_path: str = 'configurations',
    ):
        super().__init__(filename, file_path)

        t0 = (
            self.all_params['sequence']['tau'] 
            + 1/(2*self.all_params['pulse']['rabi_freq'])
        )

        self.standard_seq_duration = (
            self.all_params['sequence']['baseline_length'] 
            * t0
        ) # assuming all pulses are pi-rotations in the sequence

    def set_sequence(self, input: Optional[Any]) -> None:
        """
        Set the sequence based on the provided input.

        Args:
            input: The sequence input (None, str, List[int], or List[str]).

        Raises:
            ValueError: If the input format is invalid.
        """

        self.seq.set_pulses(input)

        num_cycle = (
            self.all_params['sequence']['baseline_length'] 
            / len(self.seq.seq_labels)
        )
        if not num_cycle.is_integer():
            raise ValueError(
                "num_cycle must be an integer. "
                "Please adjust 'baseline_length' or sequence length."
            ) 

        self.num_cycle = int(num_cycle)



    def get_performance_metrics(
            self, 
            repetitions: int = 500,
            objective: str = None
        ):
        """
        Run simulation and calculate scores for all objective functions.

        Args:
            repetitions: Number of times to repeat the simulation.

        Returns:
            Dictionary of objective function scores.
        """
        training_params = self.seq.params['training']
        min_N = training_params['min_N']
        step_N = training_params['step_N']
        num_points = training_params['num_points']


        tau = self.all_params['sequence']['tau']

        t_pulse = (
            sum_angles(self.seq.seq_labels) 
            / (2 * np.pi * self.all_params['pulse']['rabi_freq'])
        ) * self.num_cycle

        t_total = tau * self.all_params['sequence']['baseline_length']  + t_pulse


        deviation = self.standard_seq_duration - t_total

        tau = tau + deviation / self.all_params['sequence']['baseline_length']

        results = self.run(
            repetitions, min_N, step_N, num_points, tau, is_training=True
        )

        results_for_training = results['avg_over_axes']['mean']
        objective_functions = get_all_objective_functions()

        if objective is None:
            return {
                name: func(results_for_training, training_params)
                for name, func in objective_functions.items()
            }, results
        else:
            if objective not in objective_functions:
                raise ValueError(
                    f"Objective function {objective} not found."
                )
            return objective_functions[objective](
                    results_for_training, training_params
                ), results


    def run(
        self,
        repetitions: int = 1000,
        min_N: int = 1,
        step_N: int = 1,
        num_points: int = 100,
        tau: float = None,
        is_training: bool = False,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, List[float]]]:


        if tau is None: 
            # if tau is not provided, use the default value 
            # given in the config file,
            # this is just for plotting its decay curve
            tau = self.all_params['sequence']['tau']

        t_pulse = (
            sum_angles(self.seq.seq_labels) 
            / (2 * np.pi * self.all_params['pulse']['rabi_freq'])
        )


        t_total = (
            tau * self.all_params['sequence']['baseline_length']  
            + t_pulse * self.num_cycle
        ) # total duration of the sequence


        N_list = [min_N + i * step_N for i in range(num_points)]

        
        timesteps = [N * t_total for N in N_list]

        # Initialize results array
        results = np.zeros((repetitions, len(AXES), num_points))
        
        for i in range(repetitions):
            results[i] = self._run_single_round(tau, N_list)

        # Calculate statistics
        mean = np.mean(results, axis=0)
        sem = stats.sem(results, axis=0)
        interval = sem * stats.t.ppf((1 + confidence_level) / 2, repetitions - 1)
        
        lower_ci = mean - interval
        upper_ci = mean + interval
        
        # Prepare the output dictionary
        aggregated_results = {
            'timesteps': timesteps,
            'avg_over_axes': {
                'mean': np.mean(mean, axis=0).tolist(),
                'lower_ci': np.mean(lower_ci, axis=0).tolist(),
                'upper_ci': np.mean(upper_ci, axis=0).tolist()
            }
        }
        
        for i, axis in enumerate(AXES):
            aggregated_results[axis] = {
                'mean': mean[i].tolist(),
                'lower_ci': lower_ci[i].tolist(),
                'upper_ci': upper_ci[i].tolist()
            }
        
        return aggregated_results


    def _run_single_round(
        self, 
        tau,
        N_list,
    ) -> np.ndarray:
        """
        Run a single round of the simulation for all timesteps.

        Args:
            timesteps: Array of timesteps to simulate.

        Returns:
            2D array of results for each axis and timestep.
        """
        h_free, unitaries = self.sample_hamiltonian_and_pulse_error()
        results = np.zeros((len(AXES), len(N_list)))
            

        # Start of Selection
        compen_1Q = np.linalg.matrix_power(self.seq.unitary.dag().full(), self.num_cycle)


        U0 = self._calculate_total_unitary_fixed_tau(tau, h_free, unitaries)

        U0_powers = efficient_unitary_powers(U0, N_list)
        compen_powers = efficient_unitary_powers(compen_1Q, N_list)

        
        for i, N in enumerate(N_list):
            U = U0_powers[N]

            compen = reduce(np.kron, [
                compen_powers[N] for _ in range(self.ens.num_spins)
            ])

            U_temp = compen @ U
            for j, axis in enumerate(AXES):
                results[j, i] = self._calculate_avg_fidelity(U_temp, axis)
            
        return results



    def _calculate_total_unitary_fixed_tau(
        self,
        tau: float,
        h_free: np.ndarray,
        all_noisy_unitaries: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Run the simulation for a fixed tau value.

        Args:
            tau: The fixed tau value.
            h_free: The free Hamiltonian of the system.
            all_noisy_unitaries: 
                Dictionary of sequence and compensation unitaries.

        Returns:
            A total noisy unitary for the given tau value.
        """
        U = self.ens.identity.full()
        u_free = expm(-1j * tau * h_free)
        for label in self.seq.seq_labels:
            U = u_free @ U
            if label != 'null':
                U = all_noisy_unitaries['sequence'][label] @ U
                

        U_total = np.eye(U.shape[0])
        for _ in range(self.num_cycle):

            U_total = U @ U_total

        return U_total


def sum_angles(angle_list):
    total = Fraction(0)
    # Updated pattern to capture the angle part after each comma
    pattern = r'[+-][XY],(\d*pi(?:/\d+)?)'
    
    for item in angle_list:
        # Split the item by ';' to handle multiple rotations in a single string
        rotations = item.split(';')
        for rotation in rotations:
            angles = re.findall(pattern, rotation)
            for angle in angles:
                if angle == 'pi':
                    total += Fraction(1)
                elif '/' in angle:
                    numerator, denominator = angle.replace('pi', '').split('/')
                    numerator = int(numerator) if numerator else 1
                    total += Fraction(numerator, int(denominator))
                else:
                    total += Fraction(int(angle.replace('pi', '')))
    
    # Evaluate the fraction to a floating-point number by multiplying with pi
    return float(total) * np.pi


def efficient_unitary_powers(U, exponents):
    # Sort exponents in ascending order
    sorted_exp = sorted(set(exponents))
    max_exp = sorted_exp[-1]
    
    # Initialize result dictionary and powers cache
    results = {}
    powers = {1: U}
    
    # Compute powers of 2 up to the largest needed
    power_of_2 = 1
    while power_of_2 * 2 <= max_exp:
        power_of_2 *= 2
        powers[power_of_2] = np.matmul(powers[power_of_2 // 2], powers[power_of_2 // 2])
    
    # Compute each required power
    for exp in sorted_exp:
        if exp in powers:
            results[exp] = powers[exp]
        else:
            # Find the largest power of 2 less than or equal to exp
            largest_power = 1 << (exp.bit_length() - 1)
            
            # Multiply by additional powers as needed
            result = powers[largest_power]
            remaining = exp - largest_power
            while remaining:
                largest_power = 1 << (remaining.bit_length() - 1)
                result = np.matmul(result, powers[largest_power])
                remaining -= largest_power
            
            results[exp] = result
            powers[exp] = result  # Cache for potential future use
    
    # Return results in the order of the input list
    return {exp: results[exp] for exp in exponents}