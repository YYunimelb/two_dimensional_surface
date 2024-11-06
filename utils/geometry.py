# utils/geometry.py
import numpy as np

def find_central_atom(positions, lattice_vectors):
    """找到结构的中心原子"""
    center = 0.5 * (lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2])
    distances = np.linalg.norm(positions - center, axis=1)
    central_atom_index = np.argmin(distances)
    return central_atom_index

def find_equivalent_atoms(central_atom_index, atomic_types, total_atoms, layers, target_layer):
    """查找等价原子"""
    num_types = len(atomic_types)
    potential_equivalents = [(central_atom_index + i * num_types) for i in range(total_atoms // num_types)]
    equivalent_atoms = [index for index in potential_equivalents if index < total_atoms and layers[index] == target_layer]
    return equivalent_atoms

def are_points_collinear(positions):
    """检查点是否共线"""
    if len(positions) < 3:
        return True
    base_point = positions[0]
    base_vector = np.array(positions[1]) - np.array(base_point)
    base_vector = base_vector / np.linalg.norm(base_vector)
    for point in positions[2:]:
        current_vector = np.array(point) - np.array(base_point)
        current_vector = current_vector / np.linalg.norm(current_vector)
        cross_product = np.cross(base_vector, current_vector)
        if np.linalg.norm(cross_product) > 1e-5:
            return False
    return True

