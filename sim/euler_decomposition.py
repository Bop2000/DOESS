import numpy as np
import qutip as qt
from scipy.linalg import expm

# see chapter 4.1 of reference at https://github.com/gecrooks/on_gates

def decompose_rotation(U, euler_type='xyx'):
    if euler_type == 'xyx':
        u = np.cos(np.pi/4) + 1j * np.sin(np.pi/4) * qt.sigmay()
        udag = np.cos(np.pi/4) - 1j * np.sin(np.pi/4) * qt.sigmay()
    elif euler_type == 'zyz':
        u = qt.qeye(2)
        udag = qt.qeye(2)
    else:
        raise ValueError("Invalid Euler angle type")

    V = np.dot(u.full(), np.dot(U, udag.full()))

    alpha, theta0, theta1, theta2 = decompose_rotation_ZYZ(V)

    return alpha, theta0, theta1, theta2

def decompose_rotation_ZYZ(U):
    # Ensure U is unitary
    if not np.allclose(np.dot(U, U.conj().T), np.eye(2), atol=1e-8):
        raise ValueError("Input matrix is not unitary")

    # Extract the phase
    alpha = np.angle(np.linalg.det(U)) / 2
    V = np.exp(-1j * alpha) * U

    # Calculate theta (Y rotation)
    V00 = V[0, 0]
    V01 = V[0, 1]
    
    # Use arctan2 for better numerical stability
    theta1 = 2 * np.arctan2(np.abs(V01), np.abs(V00))

    # Calculate phi and psi (Z rotations)
    if np.isclose(np.cos(theta1 / 2), 0, atol=1e-8):
        summation = 0
    else:
        summation = 2 * np.angle(V[1, 1] / np.cos(theta1/2))
    
    if np.isclose(np.sin(theta1 / 2), 0, atol=1e-8):
        difference = 0
    else:
        difference = 2 * np.angle(V[1, 0] / np.sin(theta1/2))
    
    theta0 = (summation - difference) / 2
    theta2 = (summation + difference) / 2 

    return alpha, theta0, theta1, theta2

def compose_rotation(alpha, theta0, theta1, theta2, euler_type='xyx'):
    """Compose rotation matrix from ZYZ angles and global phase."""
    if euler_type == 'xyx':
        R1 = (-1j * theta0 * qt.sigmax() / 2).expm()
        Ry = (-1j * theta1 * qt.sigmay() / 2).expm()
        R2 = (-1j * theta2 * qt.sigmax() / 2).expm()
    elif euler_type == 'zyz':
        R1 = (-1j * theta0 * qt.sigmaz() / 2).expm()
        Ry = (-1j * theta1 * qt.sigmay() / 2).expm()
        R2 = (-1j * theta2 * qt.sigmaz() / 2).expm()

    U = np.exp(1j * alpha) * R2 * Ry * R1

    return U.full()

if __name__ == "__main__":
    import time
    import random
    from tqdm import tqdm
    start_time = time.time()
    num_tests = 10000
    
    for _ in tqdm(range(num_tests), desc="Testing decompositions", unit="test"):
        # Create a random unitary matrix
        U = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
        U, _ = np.linalg.qr(U)  # Ensure it's unitary
        
        euler_type = random.choice(['xyx', 'zyz'])
        # Decompose the matrix
        alpha, theta0, theta1, theta2 = decompose_rotation(U, euler_type)
        
        # Reconstruct the matrix to verify
        U_reconstructed = compose_rotation(
            alpha, theta0, theta1, theta2, euler_type
        )
        
        difference = U - U_reconstructed
        abs_difference = np.abs(difference)
        
        if not np.allclose(U, U_reconstructed, atol=1e-8):
            print('\nDecomposition is wrong!')
            print('-' * 50)
            print("Original matrix:")
            print(U)
            print('-' * 50)
            print("\nDecomposition angles:")
            print(f"alpha (global phase): {alpha}")
            print(f"theta0 (first Z rotation): {theta0}")
            print(f"theta1 (Y rotation): {theta1}")
            print(f"theta2 (second Z rotation): {theta2}")
            print('-' * 50)
            print("\nReconstructed matrix:")
            print(U_reconstructed)
            print('-' * 50)
            print("\nDifference between original and reconstructed:")
            print(difference)
            print(abs_difference)
            break
    else:
        print(f"\nAll {num_tests} tests on random unitaries passed.")
        average_time = (time.time() - start_time) / num_tests
        print(f"--- Each round uses {average_time:.6f} seconds ---")