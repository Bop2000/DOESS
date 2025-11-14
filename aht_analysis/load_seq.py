# %%
import csv
import ast
import numpy as np

from euler_decomposition import decompose_rotation


def get_compensation_pulses(num_repetitions, unitary):
    u_total = reduce(np.matmul, [unitary for _ in range(num_repetitions)])
    u_comp = u_total.conj().T
    
    alpha, angle0, angle1, angle2 = decompose_rotation(u_comp)

    return [
        {'angle': angle0, 'theta': 0},
        {'angle': angle1, 'theta': np.pi/2},
        {'angle': angle2, 'theta': 0},
        {'u_total': u_total, 'u_comp': u_comp}
    ]
    

def str_to_unitary(unitary_str):
    # Remove the outer quotes and newline
    unitary_str = unitary_str.strip('"').strip()
    
    # Split into rows
    rows = unitary_str.strip('[]').split('\n')
    
    # Process each row
    matrix = []
    for row in rows:
        # Clean up the row string
        row = row.strip('[] ')
        # Split the row into elements and convert each to complex
        elements = []
        for element in row.split():
            if '+0.j' in element or '-0.j' in element:
                # Handle real numbers with zero imaginary part
                real_part = float(element.replace('+0.j', '').replace('-0.j', ''))
                elements.append(complex(real_part, 0))
            else:
                # Handle other complex numbers if needed
                elements.append(complex(element))
        matrix.append(elements)
    
    return np.array(matrix)

def read_sorted_sequences(file_path):
    sorted_sequences = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for i, row in enumerate(reader):
            try:
                # Convert string representations back to lists
                row['sequence'] = ast.literal_eval(row['sequence'])
                row['seq_labels'] = ast.literal_eval(row['seq_labels'])
                row['compensations'] = ast.literal_eval(row['compensations'])
                
                # Convert score and occurrence back to appropriate types
                row['score'] = float(row['score'])
                row['occurrence'] = int(row['occurrence'])
                
                # # Convert unitary string to numpy array
                # if 'unitary' not in row:
                #     print(f"Warning: No unitary field in row {i}")
                #     continue
                    
                # # row['unitary'] = str_to_unitary(row['unitary'])
                
                sorted_sequences.append(row)
                
            except Exception as e:
                print(f"Error processing row {i}: {str(e)}")
                print(f"Row content: {row}")
                continue
    
    return sorted_sequences
