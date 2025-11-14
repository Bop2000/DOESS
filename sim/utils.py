import qutip as qt
import numpy as np
from scipy.linalg import expm
import scipy.stats as stats
import itertools
from typing import Dict, List, Tuple, Union

#-----------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------


def decompose_unitary_spherical(U):
    # Calculate the angle (rotation angle)
    trace_U = np.trace(U)
    trace_U_real = np.real(trace_U)
    
    # Ensure the value is within the valid range for arccos to avoid numerical issues
    cos_value = trace_U_real / 2
    cos_value = np.clip(cos_value, -1.0, 1.0)
    angle = 2 * np.arccos(cos_value)
    
    # Calculate rotation axis
    if not np.isclose(angle, 0, atol=1e-8):
        sin_half_angle = np.sin(angle / 2)
        nx = -np.imag(U[1, 0]) / sin_half_angle
        ny = np.real(U[1, 0]) / sin_half_angle
        nz = -np.imag(U[0, 0]) / sin_half_angle
        
        # Normalize the vector (nx, ny, nz)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        if norm > 0:
            nx, ny, nz = nx / norm, ny / norm, nz / norm

        # Convert to spherical coordinates
        phi = np.arccos(nz)  
        theta = np.arctan2(ny, nx)
    else:
        # If angle is close to 0, U is close to identity
        # We choose a default orientation
        phi = 0
        theta = 0
    
    return angle, phi, theta


# Calculate the quantum process fidelity

def compute_process_fidelity(
        num_spins: int, 
        applied_operator: np.array, 
        target_operator: np.array = None
    ) -> float:

    dimension = 2 ** num_spins
    
    if target_operator is None:
        target_operator = np.eye(dimension, dtype=complex)
    
    trace_value = np.trace(
        np.dot(np.conj(target_operator).T, applied_operator)
    )
    fidelity = np.abs(trace_value) ** 2 / dimension ** 2
    
    return fidelity

# computing statiscs of data
def compute_statistics(data: Dict[str, List[float]]) \
            -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:

        avg_scores, ci_scores = {}, {}

        for key in data.keys():
            avg_scores[key] = np.mean(np.array(data[key]), axis=0)
            ci_scores[key] = stats.norm.interval(
                0.95, 
                loc=avg_scores[key], 
                scale=stats.sem(np.array(data[key]), axis=0)
            )
        return avg_scores, ci_scores


# Generating of summation of Pauli operators in a multi-spin system
def sum_paulis(pauli_type: str, num_spins: int, 
                            for_spin: bool = False) -> qt.Qobj:
    pauli_type = pauli_type.upper()
    assert pauli_type in ['X', 'Y', 'Z'], (
        'pauli_type must be "X", "Y", or "Z"'
        )
    
    # Initialize with a zero matrix of appropriate size
    op = qt.qzero([2] * num_spins) 

    for spin_index in range(num_spins):
        pauli_operator = multi_paulis(
            coords=[spin_index], 
            paulis=[pauli_type], 
            num_spins=num_spins,
            for_spin=for_spin
        )
        op += pauli_operator
    
    return op

# Generating multi-qubit Pauli operators as qutip objects
def multi_paulis(
        coords: List[int], 
        paulis: List[str], 
        num_spins: int,
        for_spin: bool = True,
    ) -> qt.Qobj:
    '''
    Generate a desired multi-qubit Pauli string for a system of N spins.
    
    Parameters:
    - coords (List[int]): The coordinates of the spins.
    - paulis (List[str]): The Pauli operators on the corresponding spins.
    - num_spins (int): The number of spins in the system.
    - for_spin (bool): If True, add the conventional factor of 1/2 to Paulis. 
                        We assume a system of spin-1/2 particles. 
    
    Returns:
    - qt.Qobj: The corresponding Pauli operator as a QuTiP object.
    '''
    paulis = [p.upper() for p in paulis]  # Convert to uppercase
    assert all(0 <= c <= num_spins - 1 for c in coords), (
        "Each coordinate must be within the range of num_spins"
    )

    assert all(i != j for i, j in itertools.combinations(coords, 2)), (
        "Each coordinate must be unique"
    )

    assert all(p in ['X', 'Y', 'Z'] for p in paulis), (
        "Each Pauli operator must be 'X', 'Y', or 'Z'"
    )

    if len(coords) != len(paulis):
        raise ValueError(
            "Number of coordinates and number of Paulis do not match."
        )

    pauli_string = ""
    for i in range(num_spins):
        if i in coords:
            index = coords.index(i)
            pauli_string += paulis[index]
        else:
            pauli_string += 'I'
    
    return str_to_op(pauli_string, for_spin)


def str_to_op(pauli_str: str, for_spin: bool = True) -> qt.Qobj:
    valid_chars = {'I', 'X', 'Y', 'Z'}
    pauli_str = pauli_str.upper()  # Convert to uppercase
    if not set(pauli_str).issubset(valid_chars):
        raise ValueError("Invalid characters in the input list of strings.")
    if for_spin:
        spin_operators = {
            'I': qt.qeye(2),
            'X': qt.sigmax()/2,
            'Y': qt.sigmay()/2,
            'Z': qt.sigmaz()/2
        }
    else:
        spin_operators = {
            'I': qt.qeye(2),
            'X': qt.sigmax(),
            'Y': qt.sigmay(),
            'Z': qt.sigmaz()
        }
    return qt.tensor([spin_operators[p] for p in pauli_str])


#------------------------------------------
# Functions for generating propagator
#--------------------------------------------
def generate_propagator(h: qt.Qobj, t: float, n: int) -> np.ndarray:
    if h.shape[0] != 2**n:
        raise ValueError(
            f"Hamiltonian dimension {dim} does not match {n} qubits"
        )
    if n <= 5: 
        #spectral-decomposition method is more efficient in these cases
        return generate_propagator_spectral(h, t)
    else:
        return generate_propagator_direct(h, t)

def generate_propagator_spectral(h: qt.Qobj, duration: float) -> np.ndarray:

    eigenvalues, eigenvectors = np.linalg.eigh(h.full())
    exp_Lambda = np.exp(-1j * duration * eigenvalues)
    return (eigenvectors * exp_Lambda) @ eigenvectors.conj().T

def generate_propagator_direct(h: qt.Qobj, duration: float) -> np.ndarray:
    return expm(-1j * duration * h.full())