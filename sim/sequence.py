import json
from typing import Dict, List, Tuple, Union, Optional
import re
import numpy as np
import qutip as qt
import random
import math
from functools import reduce

from . import utils
from . import euler_decomposition as euler
from . import register
from . import awg

from aht_analysis.aht_system import aht_disorder
from aht_analysis.aht_system import aht_dipolar
from aht_analysis.aht_rotation import aht_rotation
from aht_analysis.aht_finite import aht_finite


# Load preset sequences from JSON file
with open('preset_sequences.json', 'r') as file:
    PRESET_SEQUENCES = json.load(file)['preset_sequences']

class Sequence:
    def __init__(self, all_params: Dict, labels:Dict, candidates:Dict):
        """Initialize a Sequence instance."""
        self.params = all_params

        # free decay by default
        self.seq_labels = ['null']
        # Dict of all basic pulses that are used in the sequence
        self.used_pulses = self._generate_used_pulses()

        self.all_basic_labels = labels
        self.all_candidates = candidates

        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.I = np.eye(2, dtype=np.complex128)


    def set_pulses(
        self, input: Optional[Union[str, List[int], List[str]]] = None
    ):
        """
        Set the pulse sequence based on the provided input,
        and update the Dict of all used basic pulses

        Args:
            input: Pulse sequence input. 
            Can be None, str, List[int], or List[str].

        Raises:
            ValueError: If the input format is invalid.
        """
        if isinstance(input, str) and input in PRESET_SEQUENCES:
            self.seq_labels = self._get_preset_sequence(name=input)

        elif isinstance(input, list) and \
                all(isinstance(item, int) for item in input):
            self.seq_labels = self._get_candidate_sequence(indices=input)

        elif isinstance(input, list) and \
                all(isinstance(item, str) for item in input):
            self.seq_labels = input
        else:
            raise ValueError(self._get_error_message(input))
        # Update the Dict of all pulses that are used in the sequence
        self.used_pulses = self._generate_used_pulses()
    
    def _get_error_message(self, input) -> str:
        return (
            f"Invalid input provided: {input}\n"
            "Accepted input formats are:\n"
            "  1. None, []: Set free decay without any pulses.\n"
            "  2. A string representing the name of a preset sequence:"
            "       e.g., 'droid47', 'cory'.\n"
            "  3. A list of integers: preset candidates:"
            "       e.g., [1] for Hahn echo.\n"
            "  4. A list of strings: composite basic pulses: \n"
            "       e.g., ['+X,pi/2;-Y,pi/2', '+X,pi/2;-Y,pi/2;+X,pi/2'].\n"
        )


    def calculate_aht_score(self) -> Tuple[float, float]:
        """Calculate the AHT score for the sequence."""
        num_cycle=2
        
        s1 = aht_disorder(self.seq_labels, num_cycle=num_cycle)[0]
        s3 = aht_dipolar(self.seq_labels, num_cycle=num_cycle)[0]
        s5 = aht_rotation(self.seq_labels, num_cycle=num_cycle)[0]

        H = self.Z
        s2 = aht_finite(self.seq_labels, H, num_cycle=num_cycle)[0]

        H = np.kron(self.X, self.X) + np.kron(self.Y, self.Y) - np.kron(self.Z, self.Z)
        s4 = aht_finite(self.seq_labels, H, num_cycle=num_cycle)[0]

        return s1,s2,s3,s4,s5
    
    def calculate_aht_s1_s2_s3(self) -> Tuple[float, float]:
        """Calculate the AHT score for the sequence."""
        num_cycle=2
        
        s1 = aht_disorder(self.seq_labels, num_cycle=num_cycle)[0]
        s3 = aht_dipolar(self.seq_labels, num_cycle=num_cycle)[0]

        H = self.Z
        s2 = aht_finite(self.seq_labels, H, num_cycle=num_cycle)[0]

        return s1,s2,s3
    
    def calculate_aht_s4_s5(self) -> Tuple[float, float]:
        """Calculate the AHT score for the sequence."""
        num_cycle=2
        
        H = np.kron(self.X, self.X) + np.kron(self.Y, self.Y) - np.kron(self.Z, self.Z)
        s4 = aht_finite(self.seq_labels, H, num_cycle=num_cycle)[0]

        s5 = aht_rotation(self.seq_labels, num_cycle=num_cycle)[0]

        return s4,s5

    def sample_unitaries_with_pulse_errors(
        self, ensemble:'Ensemble'
    ) -> Dict[str, Dict[str, qt.Qobj]]:
        """Sample pulse errors and return unitaries for the sequence."""
        errors = {
            'duration': self.pulse_errors['duration'] * np.random.randn(),
            'phase': self.pulse_errors['phase'] * np.random.randn(),
        }
        

        # first generate all noisy basic pules used in the sequence
        noisy_used_pulses = {
            label: pulse.add_errors(errors).unitary(ensemble)
            for label, pulse in self.used_pulses.items()
        }
        # then calculate the unitaries for these noisy pulses
        noisy_seq_unitaries = {} 
        noisy_seq_unitaries = {
            label: label_to_unitary(ensemble, label, noisy_used_pulses)
            for label in self.seq_labels
            if label not in noisy_seq_unitaries
        }


        # calculate the noisy compensating unitaries
        pulse = self.compensating_pulses[0]
        noisy_compen_unitary = pulse.add_errors(errors).unitary(ensemble)
        for pulse in self.compensating_pulses[1:]:
            u = pulse.add_errors(errors).unitary(ensemble)
            noisy_compen_unitary = u @ noisy_compen_unitary
  

        return {
            'sequence': noisy_seq_unitaries, # Dict[str, np.ndarray]
            'compensation': noisy_compen_unitary # List[np.ndarray]
        }

    def _get_preset_sequence(self, name: str) -> List[str]:
        """Get a preset sequence by name."""
        if name not in PRESET_SEQUENCES:
            available_presets = ", ".join(PRESET_SEQUENCE.keys())
            raise ValueError(
                f"Preset sequence '{name}' not found. "
                f"Available presets:\n{available_presets}"
            )
        return PRESET_SEQUENCES[name]

    def _get_candidate_sequence(self, indices: List[int]) -> List[str]:
        """Get a sequence from candidate indices."""
        
        return [
            create_composite_pulse_label([
                self.all_basic_labels[i] for i in self.all_candidates[index]
            ])
            for index in indices
        ]

    def _generate_used_pulses(self) -> Dict[str, awg.Pulse]:
        """Generate all pulses used in the sequence."""
        all_used_pules = {}
        all_used_pules = {
            divided_label: awg.Pulse(
                rabi=self.rabi_freq, pulse_str=divided_label
            )
            for label in self.seq_labels
            for divided_label in divide_pulse_label(label)
            if divided_label not in all_used_pules
        }
        return all_used_pules
    
    @property
    def aht(self) -> Dict[str, qt.Qobj]:
        """Calculate the average Hamiltonian theory (AHT) for the sequence."""
        all_disorder = [qt.sigmaz()]
        all_interaction = [sum([
            strength * utils.str_to_op(pauli)
            for pauli, strength in self.interaction_type.items()
        ])]

        uk = qt.qeye(2)
        for composite_label in self.seq_labels[:-1]:
            u_each_round = qt.qeye(2)
            for label in divide_pulse_label(composite_label):
                pulse = self.used_pulses[label]
                u_each_round = pulse.unitary(qObj=True) * u_each_round
            uk = u_each_round * uk    
            wk = qt.tensor([uk, uk])

            all_disorder.append(uk.dag() * all_disorder[0] * uk)
            all_interaction.append(wk.dag() * all_interaction[0] * wk)
        

        avg_disorder = sum(all_disorder) / (len(self.seq_labels) + 1)
        avg_interaction = sum(all_interaction) / (len(self.seq_labels) + 1)

        return {
            "disorder": avg_disorder,
            "interaction": avg_interaction,
            'int_type': self.interaction_type.items()
        }
    

    def frame_representation(self):

        def pauli_transform_map(unitary):   
            paulis = [qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()]
            transformed_paulis = []

            for pauli in paulis:
                transformed = unitary @ pauli @ unitary.conj().T
                coeffs = [0.5 * np.trace(transformed @ p).real for p in paulis]
                transformed_paulis.append(coeffs)
        
            return np.array(transformed_paulis)

        trans_mappings = []
        for composite_label in self.seq_labels:
            u_each_round = qt.qeye(2)
            for label in divide_pulse_label(composite_label):
                pulse = self.used_pulses[label]
                u_each_round = pulse.unitary(qObj=True) * u_each_round
        
            mapping = pauli_transform_map(u_each_round.full())
            trans_mappings.append(mapping)
        
        frame_orderings = []
        ordering = np.eye(3)
        for mapping in trans_mappings:
            ordering = mapping @ ordering
            frame_orderings.append(ordering)
        

        transformed_pauli_x = [np.array([1.0, 0.0, 0.0])]
        transformed_pauli_y = [np.array([0.0, 1.0, 0.0])]
        transformed_pauli_z = [np.array([0.0, 0.0, 1.0])]

        for m in frame_orderings:
            transformed_pauli_x.append(m @ transformed_pauli_x[-1])
            transformed_pauli_y.append(m @ transformed_pauli_y[-1])
            transformed_pauli_z.append(m @ transformed_pauli_z[-1])
    
        
        return {
            'frame_orderings': frame_orderings,
            'trans_mappings': trans_mappings,
            'transformed_pauli_x': transformed_pauli_x,
            'transformed_pauli_y': transformed_pauli_y,
            'transformed_pauli_z': transformed_pauli_z,
        }
   
    
    def pulse_matrices(self):
        paulis = np.array([qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()])
        
        def batched_pauli_transform(unitary):
            transformed_paulis = []
            for pauli in paulis:
                transformed = unitary @ pauli @ unitary.conj().T
                coeffs = [0.5 * np.trace(transformed @ p).real for p in paulis]
                transformed_paulis.append(coeffs)
            return np.array(transformed_paulis)
    
        trans_mappings = []
        for composite_label in self.seq_labels:
            u_total = qt.qeye(2)
            for label in divide_pulse_label(composite_label):
                u_total = self.used_pulses[label].unitary(qObj=True) * u_total
            
            trans_mappings.append(batched_pauli_transform(u_total.full()))
        
        frame_orderings = []
        ordering = np.eye(3)
        for mapping in trans_mappings:
            ordering = mapping @ ordering
            frame_orderings.append(ordering)
        return np.array(frame_orderings).reshape((len(self.seq_labels), 9))


    @property
    def interaction_type(self) -> Dict[str, float]:
        return self.params['spin']['interaction_type']

    @property
    def rabi_freq(self) -> float:
        # convert to angular frequency
        return self.params['pulse']['rabi_freq'] * (2 * np.pi)
    
    @property
    def pulse_errors(self) -> Dict[str, float]:
        return {
            'duration': self.params['pulse']['duration_error'],
            'phase': self.params['pulse']['phase_error'] * np.pi,
        }
    
    @property
    def unitary(self) -> qt.Qobj:
        """Calculate the total unitary of the sequence."""
        U = qt.qeye(2)
        for composite_label in self.seq_labels:
            for label in divide_pulse_label(composite_label):
                pulse = self.used_pulses[label]
                U = pulse.unitary(qObj=True) * U
        return U

    @property
    def compensating_pulses(self) -> List[awg.Pulse]:
        """Compute the compensation pulses for the sequence."""
        udag = self.unitary.dag()
        alpha, angle0, angle1, angle2 = euler.decompose_rotation(udag.full())
        return [
            awg.Pulse(rabi=self.rabi_freq, angle=angle0, theta=0),
            awg.Pulse(rabi=self.rabi_freq, angle=angle1, theta=np.pi/2),
            awg.Pulse(rabi=self.rabi_freq, angle=angle2, theta=0)
        ]



#------------------------------------------------------------------------------
def avg_fidelity(
    state: qt.Qobj,
    eigenvalues: np.ndarray,
    eigenstates: List[qt.Qobj],
    dimension: int
) -> float:
    """Calculate the average fidelity for a given state."""
    if dimension == 2:
        state = qt.tensor([state, state])
    def gaussian_integers(num_points, mean, std_dev):
        return [
            round(random.gauss(mean, std_dev))
            for _ in range(num_points)
        ]
    points = 500
    mean = 0
    std = np.pi
    angles = np.array(gaussian_integers(points, mean, std))
    overlaps = np.array([
        sum(abs(eigenstate.dag() * state)**2 * np.exp(-1j * angle * eigenvalue)
            for eigenstate, eigenvalue in zip(eigenstates, eigenvalues))
        for angle in angles
    ])
    return np.mean(np.abs(overlaps) ** 2)


def label_to_unitary(
    ens:'Ensemble', 
    composite_label: str, 
    used_basic_pulses: Dict[str, np.ndarray]
) -> qt.Qobj:
    """Generate the unitary for a composite pulse label on a given ensemble."""

    divided_labels = divide_pulse_label(composite_label)
    u = used_basic_pulses[divided_labels[0]]
    for label in divided_labels[1:]:
        u = used_basic_pulses[label] @ u
        
    return u



def divide_pulse_label(input_string: str) -> List[str]:
    """Divide a composite pulse label into individual pulse labels."""
    if input_string == 'null':
        return ['null']

    rotations = input_string.split(';')
    processed_rotations = []

    for rotation in rotations:
        rotation = rotation.strip()
        if not rotation:
            continue

        parts = rotation.split(',')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid rotation format: {rotation}. "
                "Expected format: '±X,angle' or '±Y,angle'"
            )

        axis_dir, angle = parts
        validate_axis_direction(axis_dir)
        validate_angle_format(angle)

        axis_dir = axis_dir[0] + axis_dir[1:].upper()
        processed_rotations.append(f"{axis_dir},{angle.strip()}")

    if not processed_rotations:
        raise ValueError("Input string is empty or contains no valid rotations")

    return processed_rotations

def validate_axis_direction(axis_dir: str):
    """Validate the axis direction of a pulse."""
    if not (axis_dir.startswith('+') or axis_dir.startswith('-')) or \
       axis_dir[1:].upper() not in ['X', 'Y']:
        raise ValueError(
            f"Invalid axis direction: {axis_dir}. "
            "Expected +X, -X, +Y, or -Y"
        )

def validate_angle_format(angle: str):
    """Validate the angle format of a pulse."""
    angle_pattern = r'^-?\d*\.?\d*(\s*\*?\s*pi\s*(/\s*\d+)?)?$'
    if not re.match(angle_pattern, angle.strip()):
        raise ValueError(f"Invalid angle format: {angle}")

def create_composite_pulse_label(labels: List[str]) -> str:
    """Create a composite pulse label from a list of basic pulse labels."""
    return ';'.join(labels)




def map_sequence_to_target(seq_in_exp1, original_config, target_config, path='configurations'):
    exp1 = Experiment(original_config, path)
    exp2 = Experiment(target_config, path)


    exp1.set_sequence(seq_in_exp1)
    seq_labels = exp1.sequence_labels[:-1]

    exp2_basics = exp2.all_basic_labels
    exp2_cand = exp2.candidates

    seq_in_exp2 = []
    for label in seq_labels:
        temp_list = [
            next(
                (key for key, value in exp2_basics.items() if value == part), None
            ) 
                for part in divide_pulse_label(label)
        ]

        key_found = next(
            (key for key, value in exp2_cand.items() if value == temp_list), None
        )

        seq_in_exp2.append(key_found)

    exp2.set_sequence(seq_in_exp2)
    if exp2.sequence_labels == exp1.sequence_labels:
        print('The sequence was mapped correctly.')
        print(seq_in_exp2)
        return seq_in_exp2
    else:
        raise ValueError('The sequence could not be mapped correctly.')