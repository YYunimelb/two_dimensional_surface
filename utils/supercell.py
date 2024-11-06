# utils/supercell.py
import numpy as np
import copy

class SupercellBuilder:
    def __init__(self, positions, lattice_vectors, atomic_types, replication=(-1, 1, -1, 1, -1, 1)):
        self.positions = positions
        self.lattice_vectors = lattice_vectors
        self.atomic_types = atomic_types
        self.replication = replication

    def create_supercell(self):
        supercell_positions = []
        supercell_atomic_types = copy.deepcopy(self.atomic_types)

        ix_start, ix_end, iy_start, iy_end, iz_start, iz_end = self.replication
        shift = (0 - ix_start) * self.lattice_vectors[0] + (0 - iy_start) * self.lattice_vectors[1] + (0 - iz_start) * \
                self.lattice_vectors[2]

        for pos in self.positions:
            supercell_positions.append(pos + shift)

        # calculate the supercell lattice vector
        supercell_lattice_vectors = np.zeros_like(self.lattice_vectors)
        supercell_lattice_vectors[0] = self.lattice_vectors[0] * (ix_end - ix_start + 1)
        supercell_lattice_vectors[1] = self.lattice_vectors[1] * (iy_end - iy_start + 1)
        supercell_lattice_vectors[2] = self.lattice_vectors[2] * (iz_end - iz_start + 1)

        for i in range(ix_start, ix_end + 1):
            for j in range(iy_start, iy_end + 1):
                for k in range(iz_start, iz_end + 1):
                    shift = (i - ix_start) * self.lattice_vectors[0] + (j - iy_start) * self.lattice_vectors[1] + (k - iz_start) * self.lattice_vectors[2]
                    if i == 0 and j == 0 and k == 0:
                        continue
                    else:
                        for pos in self.positions:
                            supercell_positions.append(pos + shift)
                        supercell_atomic_types += self.atomic_types

        return np.array(supercell_positions), supercell_atomic_types, supercell_lattice_vectors

    @staticmethod
    def calculate_distances(positions):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances
