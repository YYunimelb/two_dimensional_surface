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
    """检查点是否共面"""
    if len(positions) < 4:
        # 少于4个点一定共面
        return True

    # 选择前三个点作为基准点，计算两个基准向量
    base_point = np.array(positions[0])
    vector_1 = np.array(positions[1]) - base_point
    vector_2 = np.array(positions[2]) - base_point

    # 计算法向量
    normal_vector = np.cross(vector_1, vector_2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化法向量

    # 检查其余点是否在平面内
    for point in positions[3:]:
        vector_to_point = np.array(point) - base_point
        # 计算点到平面的垂直距离，即法向量投影
        distance = np.dot(vector_to_point, normal_vector)
        if abs(distance) > 1e-4:  # 距离接近于零表示点在平面上
            return False

    return True


def calculate_basis_vectors(supercell_positions, central_index, equivalent_indices):
    """计算两个最小的、非共线的基矢"""
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
    """找到与给定两个基矢非共面的第三个基矢"""
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
    # 原始基矢
    a_old, b_old, c_old = original_basis

    # 新基矢 a
    a_length = np.linalg.norm(a_old)
    a_new = np.array([a_length, 0, 0])

    # 新基矢 b
    b_length = np.linalg.norm(b_old)
    ab_dot = np.dot(a_old, b_old)
    b1 = ab_dot / a_length
    b2 = np.sqrt(b_length**2 - b1**2)
    b_new = np.array([b1, b2, 0])

    # 新基矢 c
    ac_dot = np.dot(a_old, c_old)
    bc_dot = np.dot(b_old, c_old)
    c_length = np.linalg.norm(c_old)
    c1 = ac_dot / a_length
    c2 = (bc_dot - b1 * c1) / b2
    c3 = np.sqrt(max(c_length**2 - c1**2 - c2**2, 0))
    c_new = np.array([c1, c2, c3])

    return np.array([a_new, b_new, c_new])
