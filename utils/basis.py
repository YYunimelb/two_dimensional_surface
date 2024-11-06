# utils/basis.py
import numpy as np

class BasisCalculator:
    def __init__(self, lattice_vectors):
        self.lattice_vectors = lattice_vectors

    def calculate_basis_vectors(self, supercell_positions, central_index, equivalent_indices):
        """根据等价原子计算 Basis vector 1 和 Basis vector 2"""
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

    def find_third_basis_vector(self, v1, v2):
        """根据已有基矢量确定第三个基矢量"""
        a, b, c = self.lattice_vectors
        search_range = range(-5, 6)
        potential_v3s = [i * a + j * b + k * c for i in search_range for j in search_range for k in search_range]

        non_coplanar_vectors = [v for v in potential_v3s if np.linalg.norm(np.dot(np.cross(v1, v2), v)) > 1e-6]

        if non_coplanar_vectors:
            v3 = min(non_coplanar_vectors, key=np.linalg.norm)
            return v3
        else:
            return None

    def compute_new_basis(self, original_basis):
        """将基矢量转换成规范化的基矢量"""
        a_old, b_old, c_old = original_basis

        a_length = np.linalg.norm(a_old)
        a_new = np.array([a_length, 0, 0])

        b_length = np.linalg.norm(b_old)
        ab_dot = np.dot(a_old, b_old)
        b1 = ab_dot / a_length
        b2 = np.sqrt(b_length**2 - b1**2)
        b_new = np.array([b1, b2, 0])

        ac_dot = np.dot(a_old, c_old)
        bc_dot = np.dot(b_old, c_old)
        c_length = np.linalg.norm(c_old)
        c1 = ac_dot / a_length
        c2 = (bc_dot - b1 * c1) / b2
        c3 = np.sqrt(max(c_length**2 - c1**2 - c2**2, 0))
        c_new = np.array([c1, c2, c3])

        return np.array([a_new, b_new, c_new])
