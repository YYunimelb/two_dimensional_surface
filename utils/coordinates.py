# utils/coordinates.py
import numpy as np

class CoordinateConverter:
    def __init__(self, basis):
        self.basis = basis
        self.inverse_basis = np.linalg.inv(basis)

    def convert_to_relative_normalize(self, supercell_positions, supercell_atomic_types):
        """将绝对坐标转换为相对坐标并进行归一化去重"""
        relative_positions = np.dot(supercell_positions, self.inverse_basis)
        normalized_positions = np.mod(relative_positions, 1)
        epsilon = 1e-3
        normalized_positions[normalized_positions > 1 - epsilon] = 0

        rounded_positions = np.round(normalized_positions, decimals=5)
        combined = np.core.records.fromarrays(
            [rounded_positions[:, 0], rounded_positions[:, 1], rounded_positions[:, 2], supercell_atomic_types],
            names='x, y, z, type')
        unique_atoms = np.unique(combined)

        return unique_atoms
