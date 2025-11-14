import qutip as qt
import itertools
from typing import Dict, List, Tuple, Union
from scipy.linalg import expm
import numpy as np
import re

from . import utils

class Pulse:
    def __init__(self, rabi, angle=None, theta=None, pulse_str=None):
        """
        Initializes a Pulse object with the given pulse parameters.

        Args:
            rabi (float): The Rabi frequency of the pulse.
            angle (float): The rotation angle
            theta (float): The phase of the pulse determining its axis.        
        """
        self.rabi = rabi

        if pulse_str is None:
            rotation_angle, axis_angle = self._normalize(angle, theta)
            self.duration = rotation_angle / rabi
            self.theta = axis_angle
        else:
            angle, theta = self._parse_pulse_string(pulse_str)
            self.duration = angle / rabi
            self.theta = theta

    @property
    def params(self) -> Dict[str, float]:
        """
        Returns a dictionary containing the pulse parameters.

        Returns:
            Dict[str, float]: A dictionary containing the pulse parameters.
        """
        return {
            "rabi_freq": self.rabi,
            "angle": self.duration * self.rabi,
            "theta": self.theta,
        }

    def unitary(self, ensemble:'Ensemble'=None, qObj=False)->np.array:
        if ensemble is None:
            angle = self.duration * self.rabi
            pauli = (np.cos(self.theta) * qt.sigmax() 
                        + np.sin(self.theta) * qt.sigmay())
            u = np.cos(angle/2) * qt.qeye(2) - 1j * np.sin(angle/2) * pauli
            if qObj:
                return u
            else:
                return u.full()
    
        else:
            X = ensemble.global_paulis['X'] / 2
            Y = ensemble.global_paulis['Y'] / 2
            Z = ensemble.global_paulis['Z'] / 2
            identity = ensemble.identity
            num_spins = ensemble.params['num_spins']


            if self.duration == 0:
                return identity.full()

            else:      
                pulse_hamiltonian = self.rabi * (
                    np.cos(self.theta) * X  + np.sin(self.theta) * Y
                ) + ensemble.hamiltonian

                return expm(-1j * pulse_hamiltonian.full() * self.duration)
    
    def add_errors(self, errors) -> 'Pulse':
        """
        Add the sampled errors to the pulse parameters and returns a new Pulse
        object. 
        """

        # duration error is a ratio
        angle = self.rabi * self.duration * (1 + errors['duration']) 

        theta = self.theta + errors['phase']

        return Pulse(self.rabi, angle, theta)


    def _parse_pulse_string(self, pulse_str):
        """
        Parse a pulse string e.g., "-Y,pi/3", "+X,2pi/3"
        
        Returns rotation_angle and axis_angle for the pulse
        """

        if pulse_str == "null":
            return (0, 0)  # Special case for null pulse

        # Split the string into direction and angle parts
        try:
            direction, angle_str = pulse_str.split(',')
        except ValueError:
            raise ValueError(f"Invalid pulse string format: {pulse_str}")

        # Determine the axis angle
        if len(direction) < 2:
            raise ValueError(f"Invalid direction format: {direction}")
        
        sign = direction[0]
        axis = direction[1].upper()

        # Determine the sign and axis
        if direction.startswith('+'):
            if axis == 'X':
                axis_angle = 0
            elif axis == 'Y': 
                axis_angle = np.pi/2

        elif direction.startswith('-'):
            if axis == 'X':
                axis_angle = np.pi
            elif axis == 'Y':
                axis_angle = -np.pi/2

        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Parse the angle
        if angle_str == 'pi':
            rotation_angle = np.pi
        elif 'pi' in angle_str:
            # Use regex to parse expressions like 'pi/3', '2pi/3', etc.
            match = re.match(r'(\d+)?pi(/(\d+))?', angle_str)
            if match:
                numerator = int(match.group(1)) if match.group(1) else 1
                denominator = int(match.group(3)) if match.group(3) else 1
                rotation_angle = numerator * np.pi / denominator
            else:
                raise ValueError(f"Unable to parse angle: {angle_str}")
        else:
            try:
                rotation_angle = float(angle_str)
            except ValueError:
                raise ValueError(f"Unable to parse angle: {angle_str}")
        
        return rotation_angle, axis_angle


    def _normalize(self, rotation_angle, axis_angle):
        """
        Normalize an angle in radians to the range [-π, π].
        
        Args:
        angle_rad (float): Angle in radians
        
        Returns:
        float: Normalized angle in radians, in the range [-π, π]
        """
        # Use modulo operation to bring the angle within [-2π, 2π]
        rotation_angle = rotation_angle % (2 * np.pi)
        
        # Adjust to [-π, π] range
        if rotation_angle > np.pi:
            rotation_angle -= 2 * np.pi
        elif rotation_angle <= -np.pi:
            rotation_angle += 2 * np.pi

        if rotation_angle <  0:
            rotation_angle = abs(rotation_angle)
            if axis_angle < 0:
                axis_angle += np.pi
            else:
                axis_angle -= np.pi
        
        return rotation_angle, axis_angle