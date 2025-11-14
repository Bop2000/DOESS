import qutip as qt
import itertools
from typing import Dict, List, Tuple, Union
from scipy.linalg import expm
import numpy as np

from . import utils

class Ensemble:

    def __init__(self, spin_params):
        self.params = spin_params

        # Precompute some useful matrices
        self.zero = self._compute_zero()
        self.identity = self._compute_identity()
        self.global_paulis = self._compute_global_paulis()
        self.pauli_eigenstates = self._compute_pauli_eigenstates()
        self.pauli_dict = self._compute_pauli_dict()

        self.hamiltonian = self.sample_hamiltonian()

    def sample_hamiltonian(self) -> qt.Qobj:
        return self.sample_disorder() + self.sample_interaction()
    
    def sample_disorder(self) -> qt.Qobj:
        """
        Generates the disorder Hamiltonian of the ensemble.
        This term includes the larmor precession.

        Returns:
            qt.Qobj: The disorder Hamiltonian of the ensemble.
        """
        hamiltonian = self.zero
        for spin_index in range(self.num_spins):
            amplitude = self.disorder_stren * np.random.randn()
            amplitude += self.detuning_stren
            hamiltonian += amplitude * self._multi_paulis(
                coords=[spin_index], paulis=['Z'], 
                num_spins=self.num_spins
            )
        return hamiltonian
    
    def sample_interaction(self) -> qt.Qobj:
        """
        Generate the interaction Hamiltonian of the ensemble.

        Returns:
            qt.Qobj: The interaction Hamiltonian of the ensemble.
        """
        positions = self._sample_spin_positions()

        interaction = self.zero
        for i, j in itertools.combinations(range(self.num_spins), 2):
            distance, alpha_ij \
                = self._calculate_pair_properties(positions[i], positions[j])

            for inter_type, relative_stren in self.interaction_type.items():
                amplitude_ij = self.J0 * alpha_ij * relative_stren / distance**3
                operator_ij = self._calculate_operator(i, j, inter_type)

                interaction += amplitude_ij * operator_ij

        return interaction

    def _sample_spin_positions(self) -> np.ndarray:
        cube_size = (self.num_spins / self.density) ** (1/3)
        return (np.random.rand(self.num_spins, 3) - 0.5) * cube_size
    
    def _calculate_pair_properties(
        self, pos1: np.ndarray, pos2: np.ndarray
    ) -> Tuple[float, float]:
        diff = pos1 - pos2 # [dx, dy, dz]
        distance = np.linalg.norm(diff) # sqrt(dx^2 + dy^2 + dz^2)
        alpha_ij = 3 * np.cos(diff[2] / distance) ** 2 - 1
        return distance, alpha_ij
    
    def _calculate_operator(
        self, i: int, j: int, inter_type: str
    ) -> qt.Qobj:
        return self._multi_paulis(
                    coords=[i, j], 
                    paulis=[inter_type[0], inter_type[1]], 
                    num_spins=self.num_spins,
                    for_spin=True #add 1/2 to Paulis   
                )
        return pair_operator

    @property
    def num_spins(self) -> int:
        return self.params['num_spins']

    @property
    def density(self) -> float:
        # convert from ppm to m^3
        DENSITY_FACTOR = 1.76e23
        return self.params['density_ppm'] * DENSITY_FACTOR
    
    @property
    def disorder_stren(self) -> float:
        # convert to angular frequency
        return self.params['disorder'] * (2 * np.pi)
    
    @property
    def interaction_type(self) -> Dict[str, float]:
        return self.params['interaction_type']

    @property
    def J0(self) -> float:
        return self.params['J0']
    
    @property
    def detuning_stren(self) -> List[float]:
        # convert to angular frequency
        return self.params['detuning'] * (2 * np.pi)

    def _multi_paulis(self, coords: List[int], paulis: List[str], 
        num_spins: int, for_spin: bool = True):
        pauli_string = ""
        for i in range(num_spins):
            if i in coords:
                index = coords.index(i)
                pauli_string += paulis[index]
            else:
                pauli_string += 'I'

        if pauli_string in self.pauli_dict:
            return self.pauli_dict[pauli_string]
        else:
            return utils.multi_paulis(coords, paulis, num_spins, for_spin)


    def _compute_identity(self) -> qt.Qobj:
        return qt.tensor([
            qt.qeye(2) for _ in range(self.num_spins)
        ]) 
    
    def _compute_zero(self) -> qt.Qobj:
        return qt.tensor([
            qt.qzero(2) for _ in range(self.num_spins)
        ])
    
    def _compute_global_paulis(self) -> qt.Qobj:
        ops = {}
        for pauli_type in ['X', 'Y', 'Z']:
            ops[pauli_type] = utils.sum_paulis(
                                pauli_type=pauli_type,
                                num_spins=self.num_spins,
                                for_spin=False
                            )
        # e.g., \sum_i \sigmax_i
        return ops
    
    def _compute_pauli_eigenstates(self) -> Dict[str, qt.Qobj]:
        eigenstates = {}

        eigenstates['X'] = qt.tensor([
                (qt.basis(2, 0) + qt.basis(2, 1)).unit()
                for _ in range(self.num_spins)
            ])

        eigenstates['Y'] = qt.tensor([
                (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
                for _ in range(self.num_spins)
            ])

        eigenstates['Z'] = qt.tensor([
            qt.basis(2, 0) for _ in range(self.num_spins)
            ])

        return eigenstates
    
    def _compute_pauli_dict(self) -> dict:
        '''
        Generate all possible 2-qubit Pauli operators for n qubits
        '''
        for_spin = True # add the conventional factor of 1/2 to Paulis.

        paulis = ['I', 'X', 'Y', 'Z']
        num_spins = self.num_spins
        pauli_product = itertools.product(paulis, repeat=num_spins)
        pauli_strings = [''.join(p) for p in pauli_product]
        
        pauli_dict = {}
        for pauli_string in pauli_strings:
            non_identity_count = sum(1 for p in pauli_string if p != 'I')
                                            
            if non_identity_count <= 2:
                pauli_dict[pauli_string] = utils.str_to_op(
                                            pauli_string, for_spin=for_spin)
        
        return pauli_dict