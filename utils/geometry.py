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


import numpy as np


def are_points_coplanar(positions):
    """check whether all points in the same plane"""
    if len(positions) < 4:
        # at least 4, otherwise must in same plane
        return True

    # Select the first three points as reference points and calculate two reference vectors.
    base_point = np.array(positions[0])
    vector_1 = np.array(positions[1]) - base_point
    vector_2 = np.array(positions[2]) - base_point

    # Calculate the normal vector.
    normal_vector = np.cross(vector_1, vector_2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化法向量

    # Check if the remaining points lie within the plane.
    for point in positions[3:]:
        vector_to_point = np.array(point) - base_point
        # calculate the distance
        distance = np.dot(vector_to_point, normal_vector)
        if abs(distance) > 1e-4:  # 距离接近于零表示点在平面上
            return False

    return True


def calculate_basis_vectors(supercell_positions, central_index, equivalent_indices):
    """Calculate the two smallest, non-collinear basis vectors."""
    central_position = supercell_positions[central_index]
    vectors = [supercell_positions[idx] - central_position for idx in equivalent_indices if idx != central_index]

    unique_vectors = {}
    for v in vectors:
        norm_v = np.linalg.norm(v)
        normalized_v = tuple(v / norm_v)
        if normalized_v not in unique_vectors or norm_v < np.linalg.norm(unique_vectors[normalized_v]):
            unique_vectors[normalized_v] = v

    sorted_vectors = sorted(unique_vectors.values(), key=np.linalg.norm)

    basis_vector_1 = sorted_vectors[0]
    basis_vector_2 = None
    for vector in sorted_vectors[1:]:
        if np.linalg.norm(np.cross(basis_vector_1, vector)) > 1e-6:
            basis_vector_2 = vector
            break

    return basis_vector_1, basis_vector_2

def find_third_basis_vector(v1, v2, original_basis):
    """Find a third basis vector that is non-coplanar with the given two basis vectors."""
    a, b, c = original_basis[0, :], original_basis[1, :], original_basis[2, :]
    search_range = range(-5, 6)
    potential_v3s = [i * a + j * b + k * c for i in search_range for j in search_range for k in search_range]

    non_coplanar_vectors = [v for v in potential_v3s if np.linalg.norm(np.dot(np.cross(v1, v2), v)) > 1e-6]

    if non_coplanar_vectors:
        v3 = min(non_coplanar_vectors, key=np.linalg.norm)
        return v3
    else:
        return None
def standardization_basis(original_basis):
    # orginal basis vecter
    a_old, b_old, c_old = original_basis

    # new a
    a_length = np.linalg.norm(a_old)
    a_new = np.array([a_length, 0, 0])

    # new b
    b_length = np.linalg.norm(b_old)
    ab_dot = np.dot(a_old, b_old)
    b1 = ab_dot / a_length
    b2 = np.sqrt(b_length**2 - b1**2)
    b_new = np.array([b1, b2, 0])

    # new c
    ac_dot = np.dot(a_old, c_old)
    bc_dot = np.dot(b_old, c_old)
    c_length = np.linalg.norm(c_old)
    c1 = ac_dot / a_length
    c2 = (bc_dot - b1 * c1) / b2
    c3 = np.sqrt(max(c_length**2 - c1**2 - c2**2, 0))
    c_new = np.array([c1, c2, c3])

    return np.array([a_new, b_new, c_new])


import numpy as np

def fill_to_new_basis(self, supercell_positions, supercell_atomic_types, new_basis):
    """Transform atomic positions into a new basis and normalize coordinates."""
    # Calculate the inverse of the new basis
    inverse_basis = np.linalg.inv(new_basis)

    # Convert absolute coordinates to relative coordinates in the new basis
    relative_positions = np.dot(supercell_positions, inverse_basis)
    normalized_positions = np.mod(relative_positions, 1)

    # Normalize coordinates within the 0-1 range to prevent floating-point issues
    epsilon = 1e-3
    normalized_positions[normalized_positions > 1 - epsilon] = 0
    rounded_positions = np.round(normalized_positions, decimals=5)

    # Combine coordinates and atomic types into a structured array and remove duplicates
    combined = np.core.records.fromarrays(
        [rounded_positions[:, 0], rounded_positions[:, 1], rounded_positions[:, 2], supercell_atomic_types],
        names='x, y, z, type'
    )
    unique_atoms = np.unique(combined)

    # Extract positions and atomic types from unique_atoms
    positions = np.vstack((unique_atoms['x'], unique_atoms['y'], unique_atoms['z'])).T
    atomic_types = unique_atoms['type'].tolist()

    return positions, atomic_types
